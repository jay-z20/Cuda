#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>



// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template <
	const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate 32
	const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory 32
	const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate 32
	const int THREAD_SIZE_Y, // height of block of C that each thread calculate 4
	const int THREAD_SIZE_X,  // width of block of C that each thread calculate 4
	const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
>
__global__ void MatrixMulCUDA6(
	float * __restrict__ A,
	float * __restrict__ B,
	float * __restrict__ C,
	const int K,
	const int N) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x; // [0-7]
	int ty = threadIdx.y; // [0-7]

						  // size of thread block
	const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;  // 8 x ά�� 8��
	const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y; // 8 y ά�� 8��
	const int THREAD_NUM_PER_BLOCK = bszy * bszx; // 64 һ�� block 64 ���߳�

												  // thread id
	const int tid = ty * bszx + tx; // �߳��� block �е� id [0-63]

									// shared memory

	__shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; // avoid bank conflict 32x32
	__shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N]; // 32x32
													 // registers for C
	float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = { 0 }; // 4x4 һ���̴߳����С����
													   // registers for A and B
	float frag_a[THREAD_SIZE_Y]; // 4 һ���̴߳��� As �е� 4��
	float frag_b[THREAD_SIZE_X]; // 4 һ���̴߳��� Bs �е� 4��

								 // threads needed to load one row of tile
								 // / 4 is because float4 is used
	const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4; // 32/4=8 һ�� tile ��������Ҫ 8 ���߳� load
	const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4; // 32/4=8

														// row number and col number that needs to be loaded by this thread
	const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW; // tid ����� As tile ������ [0-7]
	const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

	const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; // tid ����� tile �����У�ÿ���̴߳��� float4 ��Ҫ *4
	const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

	// row stride that thread uses to load multiple rows of a tile
	const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW; // 64/8=8 һ���̴߳��� 4������ 64���̴߳��� 256 �����ݣ�����tile��Ҫһ���̴߳�����
	const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW; // 64/8=8

																				// can not unroll since K can not be determined at this point
	for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) { // K �����ݷֳ� tile
																	 // load A from global memory to shared memory
#pragma unroll
		for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) { //i: 0 8 16 ... 56 
			FETCH_FLOAT4(As[A_TILE_ROW_START + i][A_TILE_COL]) = FETCH_FLOAT4(A[OFFSET(
				BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
				A_TILE_COL + tile_idx, // col
				K)]);
		}

		// load B from global memory to shared memory
#pragma unroll
		for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
			FETCH_FLOAT4(Bs[B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
				tile_idx + B_TILE_ROW_START + i, // row
				B_TILE_COL + BLOCK_SIZE_N * bx, // col
				N)]);
		}

		__syncthreads(); // �߳̿���ͬ��

						 // compute c
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE_K; ++k) {
			// load A from shared memory to register
#pragma unroll
			for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
				frag_a[thread_y] = As[ty * THREAD_SIZE_Y + thread_y][k];
			}

			// load B from shared memory to register
#pragma unroll
			for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
				FETCH_FLOAT4(frag_b[thread_x]) = FETCH_FLOAT4(Bs[k][THREAD_SIZE_X * tx + thread_x]);
			}

#pragma unroll
			for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
				for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
					accum[thread_y][thread_x] += frag_a[thread_y] * frag_b[thread_x];
				}
			}

		}
		__syncthreads();
	}

	// store back to C
#pragma unroll
	for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
#pragma unroll
		for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
			FETCH_FLOAT4(C[OFFSET(
				BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
				BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
				N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
		}
	}
}

// TODO add shuffle to enable GPU write back col