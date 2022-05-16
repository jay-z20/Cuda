
#include<stdio.h>
#include<stdlib.h>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<device_launch_parameters.h>


#define BLOCK_SIZE 16

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

template<
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate 128
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory 8
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate 128
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate 8
    const int THREAD_SIZE_X  // width of block of C that each thread calculate 8
>
__global__ void matrixMul(float* A,float* B,float* C,const int M,
    const int K,const int N){
    
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X; // 128/8=16
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y; // 128/8=16
    const int THREAD_NUM_PER_BLOCK = bszx * bszy; // 256

    // thread id
    const int tid = ty * bszx + tx;

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    // register for C
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

    float ldg_a_reg[4];
    float ldg_b_reg[4];
    // register for A and B
    float frag_a[2][THREAD_SIZE_Y];
    float frag_b[2][THREAD_SIZE_X];
    
    // 一个 tile 需要的线程数,float4 所以需要 除以 4
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4; // 8/4=2 tile A 每行需要 2个线程
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4; // 128/4=32 tile B 每行需要 32 个线程

    // row number and col number that needs to be loaded by this thread
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;  // 线程 在 tile A 加载数据的起始行
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW; // 线程 在 tile B 加载数据的起始行

    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4; // 线程在 tile A 中加载数据的列
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4; // 线程在 tile B 中加载数据的列

     // row stride that thread uses to load multiple rows of a tile
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW; // A tile 的行跨度 256/2=128
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW; // B tile 的行跨度

    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    // tid     0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 ... 31
    // x[0-7]  0 0 1 1 2 2 3 3 4 4 5  5  6  6  7  7  0  ... 7
    // y[0-3]  0 1 0 1 0 1 0 1 0 1 0  1  0  1  0  1  2  ... 3
    // 4x8 threads each warp for FFMA
    const int mma_tid_x = (warp_id % 2) * 64 + (lane_id / 2) % 8 * 4;
    const int mma_tid_y = (warp_id / 2) * 32 +  (lane_id / 16) * 8 + (lane_id % 2) * 4;

    A = &A[(BLOCK_SIZE_M * by)* K];
    B = &B[BLOCK_SIZE_N * bx];

    // Prefetching
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(A[OFFSET(
                A_TILE_ROW_START + i, // row
                A_TILE_COL , // col
                K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i] = ldg_a_reg[0];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i] = ldg_a_reg[1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i] = ldg_a_reg[2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i] = ldg_a_reg[3];
    }

    // load B from global memory to shared memory
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(B[OFFSET(
                B_TILE_ROW_START + i, // row
                B_TILE_COL , // col
                N )]);
    }

    __syncthreads();

    FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[0][0][mma_tid_y]);
    FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[0][0][mma_tid_y + 16]);

    FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[0][0][mma_tid_x]);
    FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[0][0][mma_tid_x + 32]);

   
    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        // load next tile
        if (tile_idx < K)
        {
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                FETCH_FLOAT4(ldg_a_reg[0]) = FETCH_FLOAT4(A[OFFSET(
                                            A_TILE_ROW_START + i, // row
                                            A_TILE_COL + tile_idx, // col
                                            K )]);
            }

            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                FETCH_FLOAT4(ldg_b_reg[0]) = FETCH_FLOAT4(B[OFFSET(
                        B_TILE_ROW_START + i + tile_idx, // row
                        B_TILE_COL , // col
                        N )]);
            }
        }
        int load_stage_idx = write_stage_idx ^ 1;
        int next_stage_flag = load_stage_idx;

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K; j++)
        {
            next_stage_flag = load_stage_idx;
            if (j < BLOCK_SIZE_K-1)
            {
               // load A from shared memory to register
                FETCH_FLOAT4(frag_a[(j+1)%2][0]) = FETCH_FLOAT4(As[next_stage_flag][(j+1)%BLOCK_SIZE_K][mma_tid_y]);
                FETCH_FLOAT4(frag_a[(j+1)%2][4]) = FETCH_FLOAT4(As[next_stage_flag][(j+1)%BLOCK_SIZE_K][mma_tid_y + 16]);
                // load B from shared memory to register
                FETCH_FLOAT4(frag_b[(j+1)%2][0]) = FETCH_FLOAT4(Bs[next_stage_flag][(j+1)%BLOCK_SIZE_K][mma_tid_x]);
                FETCH_FLOAT4(frag_b[(j+1)%2][4]) = FETCH_FLOAT4(Bs[next_stage_flag][(j+1)%BLOCK_SIZE_K][mma_tid_x + 32]);
            }
            
            
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        if(tile_idx < K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[0];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[0]);
            }
            
            // use double buffer, only need one sync
            __syncthreads();
            
            FETCH_FLOAT4(frag_a[0][0]) = FETCH_FLOAT4(As[write_stage_idx][0][mma_tid_y]);
            FETCH_FLOAT4(frag_a[0][4]) = FETCH_FLOAT4(As[write_stage_idx][0][mma_tid_y + 16]);

            FETCH_FLOAT4(frag_b[0][0]) = FETCH_FLOAT4(Bs[write_stage_idx][0][mma_tid_x]);
            FETCH_FLOAT4(frag_b[0][4]) = FETCH_FLOAT4(Bs[write_stage_idx][0][mma_tid_x + 32]);

            // switch
            write_stage_idx ^= 1;
        }
        
    }while(tile_idx < K);

    //store C00 block
    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + mma_tid_y + i,
        BLOCK_SIZE_N * bx + mma_tid_x,
        N)]) = FETCH_FLOAT4(accum[i][0]);
    }
    //store C01 block
    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + mma_tid_y + i,
        BLOCK_SIZE_N * bx + mma_tid_x + 32,
        N)]) = FETCH_FLOAT4(accum[i][4]);
    }
    //store C10 block
    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + mma_tid_y + i + 16,
        BLOCK_SIZE_N * bx + mma_tid_x,
        N)]) = FETCH_FLOAT4(accum[i+4][0]);
    }
    //store C11 block
    for(int i=0; i<4; i++){
      FETCH_FLOAT4(C[OFFSET(
        BLOCK_SIZE_M * by + mma_tid_y + i+ 16,
        BLOCK_SIZE_N * bx + mma_tid_x + 32,
        N)]) = FETCH_FLOAT4(accum[i+4][4]);
    }
}

void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}



bool check(float *out,float *res,const int nx,const int ny){

    for(int i=0;i<nx;i++){
        for (int j = 0; j < ny; j++){
            if (out[i*ny+j]!=res[i*ny+j]){
                printf("i: %d j:%d\n",i,j);
                printf("out: %.2f res:%.2f\n",out[i*ny+j],res[i*ny+j]);
                return false;
            }
        }
    }
   
    return true;
}


int main(int argc,char **argv){
    if (argc != 4)
    {
        printf("usage: main.exe [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));

    double flopsPerMatrixMul = 2.0 * M * N * K;

    // initialize host memory
    srand(2022);
    randomInit(h_A, M * K);
    randomInit(h_B, K * N);

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;

    float milliseconds = 0;
    float r1,r2;
    dim3 Block(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    dim3 Grid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    //
    checkCudaErrors(cudaEventRecord(start));
    matrixMul<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X>
    <<<Grid,Block>>>(d_A,d_B,d_C,M,K,N);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    printf("------ V6 ------\n");
    printf("Time: %f ms\n",milliseconds);
    r1 = (flopsPerMatrixMul)/milliseconds/1e6;
    printf("Performance (GFlop/s): %.2f\n",(flopsPerMatrixMul)/milliseconds/1e6);
    //cublas
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaEventRecord(start));


// cublasStatus_t cublasSgemm(cublasHandle_t handle,
//                            cublasOperation_t transa, cublasOperation_t transb,
//                            int m, int n, int k,
//                            const float           *alpha,
//                            const float           *A, int lda,
//                            const float           *B, int ldb,
//                            const float           *beta,
//                            float           *C, int ldc)
            // lda
            // op = CUBLAS_OP_N 时：m
            // op = CUBLAS_OP_T 时：k
            // ldb
            // op = CUBLAS_OP_N 时：k
            // op = CUBLAS_OP_T 时：n
            // ldc: m
            
    cublasSgemm (blas_handle,
            CUBLAS_OP_N,  
            CUBLAS_OP_N,  
            M,  
            N,  
            K,  
            &alpha,  
            d_B,  
            N,     
            d_A, 
            K,  
            &beta, 
            d_C,  
            N   
        );

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    checkCudaErrors(cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    cublasDestroy(blas_handle); 

    printf("------ cublas ------\n");
    printf("Time: %f ms\n",milliseconds);
    r2 = (flopsPerMatrixMul)/milliseconds/1e6;
    printf("Performance (GFlop/s): %.2f\n",(flopsPerMatrixMul)/milliseconds/1e6);
    printf("ratio:%.2f%\n",r1/r2);
    if(check(h_C,h_C1,N,M))
        printf("the ans is right\n");
    else
        printf("the ans is wrong\n");

    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
    
    return 0;
}


