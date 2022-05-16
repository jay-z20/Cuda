#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>


#define THREAD_PER_BLOCK 256

__global__ void reduce4(float* d_in,float* d_out,unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x*8 + threadIdx.x;
    if (idx >=n) return;
    
    float *idata = d_in + blockIdx.x * blockDim.x*8;
    if (idx + 7*blockDim.x < n)
    {
        int a1 = d_in[idx];
        int a2 = d_in[idx+blockDim.x];
        int a3 = d_in[idx+2*blockDim.x];
        int a4 = d_in[idx+3*blockDim.x];
        int b1 = d_in[idx+4*blockDim.x];
        int b2 = d_in[idx+5*blockDim.x];
        int b3 = d_in[idx+6*blockDim.x];
        int b4 = d_in[idx+7*blockDim.x];
        d_in[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
     __syncthreads();

    for (int stride = blockDim.x/2; stride >32; stride>>=1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
       volatile float *vmem = idata;
       vmem[tid] += vmem[tid + 32];
       vmem[tid] += vmem[tid + 16];
       vmem[tid] += vmem[tid + 8];

       vmem[tid] += vmem[tid + 4];
       vmem[tid] += vmem[tid + 2];
       vmem[tid] += vmem[tid + 1];
    }
    
    if (tid==0) d_out[blockIdx.x] = idata[0];
}


bool check(float *out,float *res,int n){
    for(int i=0;i<n;i++){
        if(out[i]!=res[i])
            return false;
    }
    return true;
}


int main(){
    const int N = 32*1024*1024;
    int block_num = N / THREAD_PER_BLOCK/4;


    float *a = (float*) malloc(N*sizeof(float));
    float *out = (float*) malloc((N/THREAD_PER_BLOCK)*sizeof(float));
    float *res = (float*)malloc((N/THREAD_PER_BLOCK)*sizeof(float));

    float *d_a, *d_out;
    cudaMalloc((void**)&d_a,N*sizeof(float));
    cudaMalloc((void **)&d_out,(N/THREAD_PER_BLOCK)*sizeof(float));

    for (size_t i = 0; i < N; i++)
    {
        a[i] = 1;
    }
    
    for (size_t i = 0; i < block_num; i++)
    {
        float cur = 0;
        for (size_t j = 0; j < THREAD_PER_BLOCK*8; j++)
        {
            cur += a[i*THREAD_PER_BLOCK +j];
        }
        res[i] = cur;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
    dim3 Grid(N/THREAD_PER_BLOCK/8,1);
    dim3 Block(THREAD_PER_BLOCK,1);
    cudaEventRecord(start);
    ///
    reduce4<<<Grid,Block>>>(d_a,d_out,N);
    cudaEventRecord(stop);
    // 等待 stop event 完成
    cudaEventSynchronize(stop);

    cudaMemcpy(out,d_out,block_num*sizeof(float),cudaMemcpyDeviceToHost);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Time: %f ms\n",milliseconds);
    printf("Bandwidth (GB/s): %f\n",(N*4 + block_num*4)/milliseconds/1e6);

    if(check(out,res,block_num))printf("the ans is right\n");
        else{
            printf("the ans is wrong\n");
            for(int i=0;i<block_num;i++){
                printf("%lf ",out[i]);
            }
            printf("\n");
        }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
    free(res);
    return 0;
}
