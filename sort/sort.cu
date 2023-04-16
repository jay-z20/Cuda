#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>
#include <vector> 
#include <algorithm>
#define N 32
using namespace std;

void randomInit(vector<int>& vec, int size)
{
    for (int i = 0; i < size; ++i)
        vec[i] = i + 1;
    random_shuffle(vec.begin(), vec.end());
}

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}



__global__ void sort(int* d_in, int* d_out)
{
    int laneId = threadIdx.x & 31;
    int v = d_in[laneId];
    for (int offset = 1; offset < 32; offset *=2){
        int value = __shfl_xor_sync(0xFFFFFFFF,v, 2*offset-1, 32);
        bool small = !(laneId & offset);
        bool s = small ? v > value : v < value;
        v = s ? value : v;

        #pragma unroll
        for (int stride = offset; stride > 0; stride /= 2) {
            int value = __shfl_xor_sync(0xFFFFFFFF,v, stride, 32);
            bool small = !(laneId & stride);
            bool s = small ? v > value : v < value;
            v = s ? value : v;
        }
    }
    d_out[laneId] = v;
}

int main(int argc,char **argv){

    int *d_data, *d_result;
    size_t bytes = sizeof(int) * N;
    vector<int> h_data(32);
    int* h_result = (int*)malloc(bytes);

    randomInit(h_data, N);
    
    printf("before: ");
    for (int i = 0; i < N; ++i){
        printf("%d ",h_data[i]);
    }
    printf("\n");

    checkCudaErrors(cudaMalloc(&d_data, bytes));
    checkCudaErrors(cudaMalloc(&d_result, bytes));
    checkCudaErrors(cudaMemcpy( d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));
    dim3 Grid(1), Block(32);
    sort<<<Grid,Block>>>(d_data,d_result);

    checkCudaErrors(cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost));
    printf("after: ");
    for (int i = 0; i < N; ++i){
        printf("%d ",h_result[i]);
    }
    printf("\n");

    cudaFree(d_data);
    cudaFree(d_result);
    
    free(h_result);

    return 0;
}
