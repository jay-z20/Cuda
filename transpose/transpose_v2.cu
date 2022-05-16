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

// out[ix][iy] = in[iy][ix]
__global__ void transposeUnroll4Row(float *out,float *in,const int nx,const int ny){
    unsigned int ix = blockDim.x * blockIdx.x*4 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * nx + ix;
    unsigned int to = ix * ny + iy;
    if (ix + 3*blockDim.x < nx && iy < ny)
    {
        out[to] = in[ti];
        out[to + ny*blockDim.x] = in[ti + blockDim.x];
        out[to + ny*2*blockDim.x] = in[ti + 2*blockDim.x];
        out[to + ny*3*blockDim.x] = in[ti + 3*blockDim.x];
    }
    
}

// out[iy][ix] = in[ix][iy]
__global__ void transposeUnroll4Col(float *out,float *in,const int nx,const int ny){
    unsigned int ix = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti = iy * ny + ix;
    unsigned int to = ix * nx + iy;
    if (ix + 7*blockDim.x < ny && iy < nx)
    {
        out[ti] = in[to];
        out[ti + blockDim.x] = in[to + nx*blockDim.x];
        out[ti + 2*blockDim.x] = in[to + nx*2*blockDim.x];
        out[ti + 3*blockDim.x] = in[to + nx*3*blockDim.x];

        out[ti + 4*blockDim.x] = in[to + nx*4*blockDim.x];
        out[ti + 5*blockDim.x] = in[to + nx*5*blockDim.x];
        out[ti + 6*blockDim.x] = in[to + nx*6*blockDim.x];
        out[ti + 7*blockDim.x] = in[to + nx*7*blockDim.x];
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
    dim3 Grid((ny+16-1)/16/8,(nx+16-1)/16);
    cudaEventRecord(start);
    /// transposeUnroll4Row  transposeUnroll4Col
    transposeUnroll4Col<<<Grid,Block>>>(d_out,d_a,nx,ny);
    cudaEventRecord(stop);
    // 等待 stop event 完成
    cudaEventSynchronize(stop);

    cudaMemcpy(out,d_out,nBytes,cudaMemcpyDeviceToHost);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Time: %f ms\n",milliseconds);
    printf("Bandwidth (GB/s): %f\n",(N*4 + N*4)/milliseconds/1e6);

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
