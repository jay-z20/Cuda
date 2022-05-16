#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>

#define INPUT_H 1024 
#define INPUT_W 1024 
#define TILE_SIZE 32

struct SOA{
    int rshift[2];
    int rdims[2];
    int rstrides[3];
    int rshapes[3];
};


void checkResult(float *hostRef, float *gpuRef, const int size, int showme)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < size; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                    gpuRef[i]);
            break;
        }

        if (showme && i > size / 2 && i < size / 2 + 5)
        {
            // printf("%dth element: host %f gpu %f\n",i,hostRef[i],gpuRef[i]);
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}


__global__ void rollKernel(const float *in, float *out, int size,int Ndims,SOA *p) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx >= size) return;

        int *rshift = p->rshift;
        int *rdims = p->rdims;
        int *rstrides = p->rstrides;
        int *rshapes = p->rshapes;

		int new_dim = 0;
		int new_idx = idx;

		for (size_t i = 0; i < Ndims; i++)
		{
			int ind = rdims[i];
			new_dim = (idx / rstrides[ind])%rshapes[ind]+rshift[i];
			//需要考虑 越界循环
			if (new_dim>=rshapes[ind]) 
				new_idx += (rshift[i] - rshapes[ind])*rstrides[ind];
			else
				new_idx += rshift[i]*rstrides[ind];
		}
		out[new_idx] = in[idx];
        
}


// 尝试使用 transpose 的优化技巧
//x 256 的线程块处理 一行 1024 的数据
__global__ void rollKernelSmem(const float *in, float *out,unsigned int size,int Ndims,SOA *p) {

        int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

        int *rshift = p->rshift;
        int *rshapes = p->rshapes;

        int row = blockIdx.x % 1024;
		if (idx >= size) return;

        __shared__ float tile[TILE_SIZE*TILE_SIZE];
        
        // 每个线程加载 4块
        int tid;
        #pragma unroll
        for (int j = 0; j < 4; j += 1){
            
            tid = idx + j*256;
            tile[(tid + rshift[1])%1024] = in[tid]; 
        }

        __syncthreads();

        if (row + rshift[0]>=rshapes[1]){
            idx  -= (rshapes[1] -1) *rshapes[2];
        }else{
            idx += rshapes[2];
        }
        #pragma unroll
        for (int j = 0; j < 4; j += 1){
            tid = idx + j*256;
            out[tid] =  tile[tid%1024];
        }
}



//x 256 的线程块处理 一行 1024 的数据
__global__ void rollKernelSmem4(const float *in, float *out,unsigned int size,int Ndims,SOA *p) {

        int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

        //int col = idx % 1024;
        int row = blockIdx.x % 1024;
		if (idx >= size) return;

        __shared__ float tile[TILE_SIZE*TILE_SIZE];
        
        // 每个线程加载 4块
        int tid;
        #pragma unroll
        for (int j = 0; j < 4; j += 1){
            
            tid = idx + j*256;
            tile[(tid + 1)%1024] = in[tid]; 
        }

        __syncthreads();

        if (row + 1>=1024){
            idx  -= 1023*1024;
        }else{
            idx += 1024;
        }
        //int col = idx % (1024*1024);
        for (int j = 0; j < 4; j += 1){
            tid = idx + j*256;
            out[tid] =  tile[tid%1024];
        }
}


bool check(float *out,float *res,const int nx,const int ny){
    for(int i=0;i<nx;i++){
        for (int j = 0; j < ny; j++){
            if (out[i*ny+j]!=res[i*ny+j]){
                return false;
            }
        }
    }
    return true;
}




int main(){

    // 测试数据 3x4x5
    unsigned int N = 3*INPUT_H*INPUT_W;
    size_t sizes = N*sizeof(float);
    float *data = (float*) malloc(sizes);
    float *out = (float*) malloc(sizes);
    float *res = (float*)malloc(sizes);

	for (int i = 0; i < N; i++)
		data[i] = (float)( rand() & 0xFF ) / 10.0f;;

    int Ndims = 2;
    // int rshift[2] = {1,1};
    // int rdims[2] = {1,2};
    // int rstrides[3] = {20,5,1};
    // int rshapes[3] = {3,4,5};
    // 设置参数
    SOA p = {
        {1,1},
        {1,2},
        {INPUT_W*INPUT_H,INPUT_W,1},
        {3,INPUT_H,INPUT_W}
    };
    size_t nBytes = sizeof(SOA);
    SOA *d_p;
    cudaMalloc((SOA**)&d_p,nBytes);
    cudaMemcpy(d_p,&p,nBytes,cudaMemcpyHostToDevice);

    float *d_a, *d_out,*d_out1;
    cudaMalloc((float**)&d_a,sizes);
    cudaMalloc((float**)&d_out,sizes);
    cudaMalloc((float**)&d_out1,sizes);
    

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_a,data,sizes,cudaMemcpyHostToDevice);
    dim3 Block1(256);
    dim3 Grid1((N+256-1)/256);

    // 先运行 rollKernel 得到结果验证 rollKernelSmem 正确性
    rollKernel<<<Grid1,Block1>>>(d_a,d_out,N,Ndims,d_p);
    cudaMemcpy(res,d_out,sizes,cudaMemcpyDeviceToHost);

    dim3 Block(256);
    dim3 Grid(3*INPUT_H);
    cudaEventRecord(start);

    rollKernelSmem<<<Grid,Block>>>(d_a,d_out1,N,Ndims,d_p);
    cudaEventRecord(stop);
    // 等待 stop event 完成
    cudaEventSynchronize(stop);

    cudaMemcpy(out,d_out1,sizes,cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Time: %f ms\n",milliseconds);
    printf("Bandwidth (GB/s): %f\n",(N*4 + N*4)/milliseconds/1e6);
    checkResult(res,out,N,1);
    if(check(out,res,INPUT_W,INPUT_H*3))
        printf("the ans is right\n");
    else
        printf("the ans is wrong\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_p);
    free(data);
    free(out);
    free(res);
    return 0;
}
