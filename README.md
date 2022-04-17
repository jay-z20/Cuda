## 矩阵相乘

![](https://latex.codecogs.com/svg.image?C_{M\times&space;N}=A_{M\times&space;K}\times&space;B_{K\times&space;N})

对于矩阵 `C` 中 紫色`tile` ![](https://latex.codecogs.com/svg.image?32\times32) ，需要矩阵 `A` 黄色区域 ![](https://latex.codecogs.com/svg.image?32\times&space;K) 和矩阵`B` 红色区域 ![](https://latex.codecogs.com/svg.image?K\times32)

<img src="./images/image1.png" title="" alt="" data-align="center">

矩阵`A` 中黄色区域划分成 ![](https://latex.codecogs.com/svg.image?BLOCK\\_SIZE\\_M\times&space;BLOCK\\_SIZE\\_K=32\times&space;32)

矩阵`B` 中红色区域划分成 ![](https://latex.codecogs.com/svg.image?BLOCK\\_SIZE\\_K\times&space;BLOCK\\_SIZE\\_N=32\times&space;32)

把这些块取到 `__shared__ memory` `As、Bs` 中

<img src="./images/image2.png" title="" alt="" data-align="center">

每个线程处理 ![](https://latex.codecogs.com/svg.image?THREAD\\_SIZE\\_Y\times&space;THREAD\\_SIZE\\_X=4\times4) 的小矩阵，需要从 `As` 中取 ![](https://latex.codecogs.com/svg.image?4\times1)，`Bs` 中取 ![](https://latex.codecogs.com/svg.image?1\times4)

![](./images/image3.png)
