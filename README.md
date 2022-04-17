## 矩阵相乘

$C_{M\times N}=A_{M\times K}\times B_{K\times N}$

对于矩阵 `C` 中 紫色`tile` $32\times32$ ，需要矩阵 `A` 黄色区域 $32\times K$ 和矩阵`B` 红色区域 $K\times 32$ 

<img src="./images/image1.png" title="" alt="" data-align="center">

矩阵`A` 中黄色区域划分成$BLOCK\_SIZE\_M\times BLOCK\_SIZE\_K=32\times 32$

矩阵`B` 中红色区域划分成 $BLOCK\_SIZE\_K\times BLOCK\_SIZE\_N=32\times 32$

把这些块取到 `__shared__ memory` `As、Bs` 中

<img src="./images/image2.png" title="" alt="" data-align="center">

每个线程处理 $THREAD\_SIZE\_Y\times THREAD\_SIZE\_X=4\times4$ 的小矩阵，需要从 `As` 中取 $4\times1$，`Bs` 中取$1\times4$

![](./images/image3.png)
