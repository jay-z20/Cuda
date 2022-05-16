
### reduce_v0

```c
if ((tid%(2*stride))==0)
    idata[tid] += idata[tid+stride];
```

按道理来说，`v0` 版本是存在线程束分化的，但是编译器优化后，并没有统计到分化的情况

```
nvcc reduce_v0.cu -o v0
Time: 8.286208 ms
Bandwidth (GB/s): 16.260998
# 查看每个线程束上执行指令数量的平均值
nvprof --metrics inst_per_warp v0.exe
  inst_per_warp Instructions per warp  380.875000
# 查看内存加载吞吐量
nvprof --metrics gld_throughput v0.exe
    gld_throughput    Global Load Throughput  5.3477GB/s

# 分支效率：未分化的分支与全部分支比值
nvprof --metrics branch_efficiency v0.exe
    branch_efficiency   Branch Efficiency     100.00%     100.00%

# nvprof 获得分支和分化分支的时间计数器
nvprof --events branch,divergent_branch v0.exe
Invocations                                Event Name         Min         Max         Avg       Total
Device "GeForce GTX 1060 6GB (0)"
    Kernel: reduce0(float*, float*, unsigned int)
          1                                    branch    26214400    26214400    26214400    26214400
          1                          divergent_branch           0           0           0           0
```

### reduce_v1

```c
 int index = 2*stride*tid;
if (index < blockDim.x)
    idata[index] += idata[index+stride];
```

`v1` 相对于 `v0`，减少了线程束分化，`inst_per_warp` 减少了一半

虽然加载的数据量不变，**猜测可能是**，由于同一线程束的工作线程多了，且访问连续的内存块，合并读取内存起到作用

```
nvcc reduce_v1.cu -o v1
Time: 5.654272 ms
Bandwidth (GB/s): 23.830126

# 查看每个线程束上执行指令数量的平均值
nvprof --metrics inst_per_warp v1.exe
  inst_per_warp Instructions per warp  142.125000

# 查看内存加载吞吐量
nvprof --metrics gld_throughput v1.exe
    gld_throughput    Global Load Throughput  10.134GB/s

# 分支效率：未分化的分支与全部分支比值
nvprof --metrics branch_efficiency v1.exe
    branch_efficiency    Branch Efficiency     100.00%
```

### reduce_v2

交错规约，初始跨度是线程块大小的一半，每次迭代减少一半

**同样的**，`v2` 相对于 `v1` 线程束访问连续内存块的更多，所以速度更快。

**但是**，使用 `nvprof --metrics gld_throughput v2.exe` 统计出问题了

```
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GTX 1060 6GB (0)"
    Kernel: reduce2(float*, float*, unsigned int)
          1                            gld_throughput                    Global Load Throughput  2.7538GB/s  2.7538GB/s  2.7538GB/s
```


```c
for (int stride = blockDim.x/2; stride >0; stride>>=1)
{
    if (tid < stride)
        idata[tid] += idata[tid+stride];
    __syncthreads();
}
```

```
nvcc reduce_v2.cu -o v2
Time: 4.919904 ms
Bandwidth (GB/s): 27.387122
```


### reduce_v3

展开的规约，更多的并发，可以帮助隐藏指令或内存延迟

```
nvcc reduce_v3.cu -o v3
Time: 1.152096 ms
Bandwidth (GB/s): 116.555616
```

### reduce_v4

在 `v3` 基础上，对线程束内的展开

```
Time: 1.021824 ms
Bandwidth (GB/s): 131.415256
```


### reduce_v5

在 `v4` 基础上，对线程块的完全展开，不过耗时没有减少

```
Time: 1.059520 ms
Bandwidth (GB/s): 126.739712
```


### reduce_v6

在 `v4` 基础上，使用了共享内存

```
Time: 0.860672 ms
Bandwidth (GB/s): 156.097568
```


### 优化技巧总结


- 合并内存访问能过提升性能，所以线程束内的线程访问的全局内存尽量跨度大一点

- 多个内存访问，延迟隐藏

```c
float a1 = d_in[idx];
float a2 = d_in[idx+blockDim.x];
float a3 = d_in[idx+2*blockDim.x];
float a4 = d_in[idx+3*blockDim.x];
sum = a1 + a2 + a3 + a4 ;
```


- 使用共享内存加速

- 线程束内的规约

```c
if (tid < 32)
{
    volatile float *vmem = smem;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];

    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
}
```

