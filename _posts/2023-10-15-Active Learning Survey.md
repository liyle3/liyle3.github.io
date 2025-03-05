---
redirect_from: /_posts/2023-10-15-Active Learning Survey.md
title: Active Learning Survey
tags:
  - 论文阅读
---



# Related work for Active Learning



## notation

* $\alpha(x, M)$：acquisition function for model $M$



## Uncertainty Based Method

### 1. Entropy：

选择使得预测分布熵最大的数据 x
$$
\alpha(x, M) = \arg\max _x \left[- \sum_k p _M [y=k|x] \log p _M(y=k|x)  \right ]
$$

### 2. LeastConfidence

选择最可能类别对应概率最小的数据 $x$
$$
\alpha(x, M) = \arg\min_x \max_{\hat y} \left[p_M(\hat y |x)  \right]
$$


### 3. Margin

最可能的两个类别在后验概率上的差异最小（最容易误判）
$$
\alpha(x, M) = \arg \min_x \left[ p_M(\hat y_1|x) - p_M(\hat y_2|x) \right]
$$



### 4. MeanSTD

所有类别的后验概率的平均标准差最大
$$
\alpha(x, M) = \arg \max _x \frac{1}{k} \sum _k \sqrt{Var_{q(w)} \left[p(y=k|x, w) \right]}
$$



### 5. **MCDropout**

* 从整体的数据中选一个子集作为初始训练集，拿来训练任务模型
* 使用训练好的模型在剩余未标注样本上以train模式跑多组预测，记录每个样本的输出
* 每个样本的多次dropout预测（先取平均再算熵） - （先算熵再求平均），得到每个样本的不确定性分数



### 5. [BALD](https://arxiv.org/pdf/1703.02910.pdf)

选择能使当前模型熵最大程度减少的数据点 $x$
$$
\arg\max _x[H(\theta|D) - E_{y \sim p(\theta|D)}[H(\theta|D, x, y)]]
$$
上式等价成求解在给定数据$D$和新增数据点$x$条件下，模型预测和模型参数之间的互信息（模型参数维度很高，直接求解比较困难）
$$
\alpha(x, M) = \arg\max _x[H(y|x, D) - E_{y \sim p(\theta|D)}[H(y|x, \theta)]]
$$



### 6. AdvDeepFool / AdvBIM

选择添加最小扰动可以使得预测类别发生改变的数据样本 $x$
$$
\alpha(x, M) = argmin_{\  r, M(x) \ne m(x + r)} - \frac{M(x+r)}{||\nabla M(x + r)||^2 _2} \nabla M(x+r),\text{(binary example)}
$$





### 7. [GAAL](https://arxiv.org/pdf/1702.07956.pdf)

使用 **GAN** 来生成靠近当前决策边界的数据

<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051937809.png" style="zoom:120%;" />

* 生成的数据样本过于靠近决策边界，oracle 可能也无法分辨样本的类别，或者z可能生成靠近边界但没意义的数据



### 8. [BGADL](https://arxiv.org/pdf/1904.11643.pdf)

使用 **GAN** 生成和池内有价值样本类似的样本，

* 使用采集函数（如**BALD**）来选择出有价值的样本集合 $S$
* 训练一个 **VAE-ACGAN** 来生成与集合 $S$ 相似的样本

 （数据增强）

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051937281.png)

### 9. LPL (Loss Prediction Loss)

添加一个模块

* 训练：最小化预测损失和目标损失之间的损失预测损失
* 推理：预测目标模型对于某个未标记输入的损失

**选择策略**：选择预测损失最高的前 $b$ 个样本





## Diversity Based Method

### 1. Kmeans

* 对未标记样本聚类 （K）
* 取最靠近簇心的样本



### 2. [CoreSet](https://arxiv.org/pdf/1708.00489.pdf)

使用模型最后一个全连接层的表征，k-center

> We use the $l_2$ distance between activations of the final fully-connected layer as the distance

$$
\alpha(x, M) = \arg \max _{x_i \in D_u} \min_{x_j \in D_l} \Delta(h(x_i), h(x_j))
$$



### 3. [Cluster-Margin](https://arxiv.org/pdf/2107.14263.pdf)

* 预处理阶段：层次聚类，每次把最接近的两个簇合并
* 采样阶段：先按照 margin scores 取 $k_m$ 个样本，得到对应的簇集合，random round-robin采样 

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051940719.png)

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051942455.png)







### 4. [VAAL](https://arxiv.org/pdf/1904.00370.pdf)

> 核心思想是用变分自编码VAE来产生一个特征，训练vae的损失包含对抗损失，让VAE尽量使标注样本、未标注样本的特征分布差不多，而判别器则试图分辨标注/无标注样本。如果判别器很确信的认为无标注样本是已标注的，说明这个无标注样本已经能很好的被现有的标注样本表示了，是个很类似的样本，反之，则说明该样本与现有标注样本差别较大，应该选来标注

* **VAE**：学习一个特征空间，用于欺骗辨别网络，使其相信所有数据均为有标签（对抗损失 + 重建损失）

* **Discriminator**：学习如何区分有标签数据和无标签数据

  

<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051943104.png" style="zoom:80%;" />

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051943809.png)



### 5. ASGN

* 基于teacher model 学习到的表征
* 选择策略：k-center



## Combined Method 

### 1. [BADGE](https://arxiv.org/pdf/1906.03671.pdf)

> **梯度范数大小表示不确定性**，和之前用熵之类的指标来表示不确定性类似，模型预测的概率小，意味着熵大，也意味着如果把这样本标了，模型要有较大的变化才能拟合好这个样本，也就是求出来的梯度大。**梯度表示多样性**，是这类方法的独特之处，用梯度向量来聚类，选到的差异大的样本就变成 让模型参数的更新方向不同的样本，而不是样本特征本身不同。

* kmeans++聚类

<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051943101.jpeg" style="zoom:80%;" />

![](https://pic4.zhimg.com/v2-2932f96ef555a1df44a6166f04c42fab_r.jpg)

### 2. [WAAL](https://arxiv.org/pdf/1911.09162.pdf)

通过采用Wasserstein距离，将AL中的交互过程建模为分布匹配

**Wasserstein距离**：衡量两个分布之间的差异，或者是最优转换代价

* DNN参数优化：有标签分布$D$与整体分布$\hat D$

* 样本选择

  * uncertainty：least confidence + uniform confidence (convex combination)

  * diversity:  1-Lipschitz 

    

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051943037.png)

### 3. DBAL

*  在当前训练集上训练一个分类器
* 获取每个未标记样本的不确定性：margin
* 按不确定分数取 $top-\beta k$个样本
* K-means，选择最接近簇心得样本



<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051943821.png" style="zoom:80%;" />













































































### 4. [Active-DPP](https://arxiv.org/pdf/1906.07975.pdf)



<img src="./figures/ADPP1.png"  />







