#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<stdlib.h>


#define BDIMY 32
#define BDIMX 32

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



// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
// 线程块 x,y = 32,8
__global__ void transposeCoalesced(float *out, const float *in,const int nx,const int ny)
{
  __shared__ float tile[BDIMY][BDIMX+1]; // 32x32
    
  int x = blockIdx.x * BDIMX + threadIdx.x;
  int y = blockIdx.y * BDIMY + threadIdx.y;
  int width =nx;
    // 每个线程加载 4块
  for (int j = 0; j < BDIMX; j += 8)
     tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*width + x]; // 写共享内存没有存储体冲突只使用了 1个事务

  __syncthreads();

  x = blockIdx.y * BDIMX + threadIdx.x;  // transpose block offset
  y = blockIdx.x * BDIMX + threadIdx.y;
  width = ny;

  for (int j = 0; j < BDIMX; j += 8)
     out[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j]; // 一个线程束
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
    dim3 Block(32,8);
    dim3 Grid((nx+32-1)/32,(ny+32-1)/32);
    cudaEventRecord(start);
    /// transposeUnroll4Row  transposeUnroll4Col
    //transposeSmem<<<Grid,Block>>>(d_out,d_a,nx,ny);
    transposeCoalesced<<<Grid,Block>>>(d_out,d_a,nx,ny);
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
