#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>

void initialData(float *in,  const int size)
{
    for (int i = 0; i < size; i++)
    {
        in[i] = (float)( rand() & 0xFF ) / 10.0f; //100.0f;
    }

    return;
}


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


void transposeHost(float *out, float *in, const int nx, const int ny)
{
    for( int iy = 0; iy < ny; ++iy)
    {
        for( int ix = 0; ix < nx; ++ix)
        {
            out[ix * ny + iy] = in[iy * nx + ix];
        }
    }
}

// 需要修改 Grid 
//  dim3 Grid((nx+16-1)/16,(ny+16-1)/16);
__global__ void transposeRow(float *out,float *in,const int nx,const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix]; // out[ix][iy] = in[iy][ix]
    }
    
}

// 需要修改 Grid 
//  dim3 Grid((ny+16-1)/16,(nx+16-1)/16);
__global__ void transposeCol(float *out,float *in,const int nx,const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < ny && iy < nx)
    {
        out[iy * ny + ix] = in[ix * nx + iy]; // out[iy][ix] = in[ix][iy]
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
    const int nx = 1024;
    const int ny = 2048;

    const size_t N = nx*ny;
    const size_t nBytes = N*sizeof(float);

    float *a = (float*) malloc(nBytes);
    float *out = (float*) malloc(nBytes);
    float *res = (float*)malloc(nBytes);

    float *d_a, *d_out;
    cudaMalloc((float**)&d_a,nBytes);
    cudaMalloc((float**)&d_out,nBytes);
    
    // 初始化矩阵
    initialData(a,N);

    transposeHost(res,a,nx,ny);
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(d_a,a,nBytes,cudaMemcpyHostToDevice);
    dim3 Block(16,16);
    dim3 Grid((nx+16-1)/16,(ny+16-1)/16);
    cudaEventRecord(start);
    /// transposeRow  transposeCol
    transposeRow<<<Grid,Block>>>(d_out,d_a,nx,ny);
    cudaEventRecord(stop);
    // 等待 stop event 完成
    cudaEventSynchronize(stop);

    cudaMemcpy(out,d_out,nBytes,cudaMemcpyDeviceToHost);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Time: %f ms\n",milliseconds);
    printf("Bandwidth (GB/s): %f\n",(N*4 + N*4)/milliseconds/1e6);
    checkResult(res,out,N,1);
    if(check(out,res,nx,ny))
        printf("the ans is right\n");
    else
        printf("the ans is wrong\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_out);
    free(a);
    free(out);
    free(res);
    return 0;
}
