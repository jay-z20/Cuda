#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>

static const int INPUT_H = 1024; //
static const int INPUT_W = 1024; //

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
		#pragma unroll
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
    int N = 3*INPUT_H*INPUT_W;
    size_t sizes = N*sizeof(float);
    float *data = (float*) malloc(sizes);
    float *out = (float*) malloc(sizes);
    float *res = (float*)malloc(sizes);

	for (int i = 0; i < N; i++)
		data[i] = i;

    int Ndims = 2;
    // int rshift[2] = {1,1};
    // int rdims[2] = {1,2};
    // int rstrides[3] = {20,5,1};
    // int rshapes[3] = {3,4,5};
    // 设置参数
    SOA p = {
        {1,1},
        {1,2},
        {1024*1024,1024,1},
        {3,1024,1024}
    };
    size_t nBytes = sizeof(SOA);
    SOA *d_p;
    cudaMalloc((SOA**)&d_p,nBytes);
    cudaMemcpy(d_p,&p,nBytes,cudaMemcpyHostToDevice);

    float *d_a, *d_out;
    cudaMalloc((float**)&d_a,sizes);
    cudaMalloc((float**)&d_out,sizes);
    

    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_a,data,sizes,cudaMemcpyHostToDevice);
    dim3 Block(256);
    dim3 Grid((N+256-1)/256);
    cudaEventRecord(start);

    rollKernel<<<Grid,Block>>>(d_a,d_out,N,Ndims,d_p);
    cudaEventRecord(stop);
    // 等待 stop event 完成
    cudaEventSynchronize(stop);

    cudaMemcpy(out,d_out,sizes,cudaMemcpyDeviceToHost);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Time: %f ms\n",milliseconds);
    printf("Bandwidth (GB/s): %f\n",(N*4 + N*4)/milliseconds/1e6);
    // checkResult(res,out,N,1);
    // if(check(out,res,nx,ny))
    //     printf("the ans is right\n");
    // else
    //     printf("the ans is wrong\n");

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
