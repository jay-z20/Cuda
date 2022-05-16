
#include<stdio.h>
#include<stdlib.h>

#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<device_launch_parameters.h>


#define BLOCK_SIZE 16

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}


__global__ void matrixMul(float* A,float* B,float* C,const int M,
    const int K,const int N){
    
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = by * blockDim.y + ty;
    int j = bx * blockDim.x + tx;

    float accu = 0.0;
    for (int k = 0; k < K; k++)
    {
        accu += A[i*M +k] * B[k*N + j];
    }
    C[i*N + j] = accu;
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

    float milliseconds = 0;
    dim3 Block(BLOCK_SIZE,BLOCK_SIZE);
    dim3 Grid(N/BLOCK_SIZE,M/BLOCK_SIZE);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    //
    checkCudaErrors(cudaEventRecord(start));
    matrixMul<<<Grid,Block>>>(d_A,d_B,d_C,M,K,N);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    printf("------ V1 ------\n");
    printf("Time: %f ms\n",milliseconds);
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
    printf("Performance (GFlop/s): %.2f\n",(flopsPerMatrixMul)/milliseconds/1e6);

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


