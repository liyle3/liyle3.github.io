---
redirect_from: /_posts/2023-07-11-MLAPP_NOTE.md
title: Notes for MLAPP
tags:
  - 学习笔记
---





# MLAPP NOTES  

` by liyle3`



## C1. Introduction

> We are drowning in information and starving for knowledge. — John Naisbitt.

### 1.1 机器学习分类

<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051718357.png" style="zoom:67%;" />

### 1.2 监督学习 or 预测类

>目标：学习一个映射 $X \rightarrow Y$
>
>训练集：$D = \{(x_i, y_i)\}_{i=1}^N$，N为训练样本个数

* **分类 or 模式识别**：$y_i \in \{1, \cdots, C\}，y_i为离散值，并且属于一个有限集合$

  * 概率预测

    > 给定输入向量 x 和训练集 D，将在可能的分类标签上的概率分布表示为 $p(y\|x,D)$
    >
    > $\sum\limits_{i=1}^{C} p(y\|x,D) = 1$
    >
    > 最大后验估计（*MAP estimate*）：$\hat y =\hat f (x)=argmax ^C_{c=1} p(y=c\|x,D)$ 

* **回归**：$y_i$ 是一个实数值



### 1.3 无监督学习 or 描述类

> 目标：在数据中探索 “interesting patterns” （knowledge discovery）
>
> 输入：$D = \{(x_i)\}_{i=1}^N$
>
> 通常没有指定模式去发掘，也没有显著有效的评估指标

从数据中发现结构，也叫知识发现（knowledge discovery）。要进行密度估计（density estimation），建立 $p(x_i\|\theta )$的模型

**监督学习和无监督学习的区别**

* 概率形式不一样，写成的是$p(x_i\|\theta )$而不是$p( y_i\|x_i, \theta )$。监督学习是条件密度估计，而无监督学习是非条件密度估计
* 无监督学习中的特征 $x_i$  是一个特征向量，要建立多元概率模型。而监督学习中的$y_i$通常是单值的，是用来去预测的



常见的无监督学习：

* 聚类分析
* 发掘隐变量
* 发掘图结构量
* 矩阵补全



### 1.4 强化学习

> 目标：在给定的奖励和惩罚的条件下，agent 学习如何行动



### 1.5 参数化模型和非参数化模型

概率模型：

* 监督学习：$p(y\|X)$
* 无监督学习：$p(X)$

**参数化模型 vs 非参数模型**：看模型是否有固定数目的参数，或者参数的数量是否随着训练集规模的增长而增长

* 参数化模型的优点是用起来更快速，而对数据分布的自然特征进行更强假设的时候就不如非参数化模型。
* 非参数化模型更加灵活，但对于大规模数据集来说在计算上比较困难。



#### 1.5.1 KNN算法：非参数化模型分类器

> 检查训练集中与输入值 x 最邻近的 K 个点，然后计算样本中每一类有多少个成员包含于这个集合中，然后返回经验分数作为估计值
>
> $p(y=c\|x,D,K) = \frac{1}{K} \sum_{i\in N_{K(x,D)}}\prod (y_i=c)$
>
> 其中，$N_{K(x,D)}$是在 D 中和点 x 最近的 K 个点的索引，$\prod (e)$是指示函数
> $$
> \prod (e) = \begin{cases} 1 & \text{if e is true}  \\
> 0 & \text{if e is false}
> \end{cases}
> $$

* **沃罗诺伊镶嵌（Voronoi tessellation，Voronoi diagram）or 狄利克雷镶嵌（Dirichlet tessellation） or 泰森多边形（Thiessen polygon）：** K = 1 的KNN分类器

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051720663.png)

* **维度诅咒**：

  > 数据点空间的边长为1，所期望的近邻空间的边长为 $e_D(f) = f ^{1/D}$，$D$ 为 维度数目，$f$ 为包含数据点的比率
  >
  > 例，维度 $D = 10$，使用数据点的比率 $f = 10\%$，则 $e_{10}(0.1) = 0.8$，近邻空间的边长为 0.8
  >
  > 维度 $D = 10$，使用数据点的比率 $f = 1\%$，则 $e_{10}(0.01) = 0.63$，近邻空间的边长为 0.63

  在得到一个比较好的距离矩阵并且有充分标签的训练集的情况下，KNN 分类器简单又好用。如果 N 趋向于无穷大，KNN 分类器的性能差不多是最佳性能的一半。但是在面对高维度输入时，KNN分类器的性能比较悲惨。

  <img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051720223.png" style="zoom:67%;" />







### 1.6 线性回归

> 回归模型：$y(x)=w^Tx + \epsilon =\sum^D_{j=1} w_j x_j + \epsilon $
>
> 其中，上式中的$w^Tx$表示的是输入向量 x 和模型的权重向量 w 的内积，$\epsilon$则是线性预测和真实值之间的残差，通常会假设残差$\epsilon$遵循高斯分布，即$\epsilon \sim N(\mu,\sigma ^2)$



为了在线性回归和高斯分布之间建立更确切的联系，可以用下面这种形式重写模型

> $p(y\|x,\theta)= N(y\|\mu(x),\sigma^2(x))$

在最简单的情况下，可以假设$\mu$是 x 的线性函数，所以$\mu =w^Tx$，而设噪音为固定的，即$\sigma^2(x) = \sigma^2$，则模型参数$\theta = (w,\sigma^2)$

对非线性关系模型，可以把线性回归中的 x 替换成某种对输入的非线性函数$\phi(x)$

> $p(y\|x,\theta) = N(y\|w^T\phi(x),\sigma^2 )$



### 1.7 逻辑回归

对于分类问题，尤其是二分类问题，可以对线性回归做两个修改：

* 对 y 不再使用高斯分布，而是换用伯努利分布，$y \in \{0, 1\}$

  > $p(y\|x,w)=Ber(y\|\mu(x))$
  >
  > 其中，$\mu(x) = E[y\|x]=p(y=1\|x)$

* 计算输入变量的一个线性组合，和之前的不同在于要通过一个函数来处理一下，以保证$0\leq \mu(x) \leq 1$

  > $\mu(x)=sigm(w^Tx)$
  >
  > 逻辑函数$sigm(\eta) = \frac{1}{1+exp(-\eta)} =\frac{e^\eta }{e^\eta +1}$

最终模型：$p(y\|x,w)=Ber(y\|sigm(w^Tx))$

这个就叫做逻辑回归，虽然看上去有点像线性回归，但实际上逻辑回归是一种分类，而并不是回归。

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051720498.png)



### 1.8 过拟合



![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051721153.png)



### 1.9 模型选择

* **误分类率**：

  > $err(f,D)=\frac{1}{N}\sum^N_{i=1}\prod(f(x_i)\neq y_i)$
  >
  > $f(x)$ 为分类器

  

* **泛化误差（generalization error）**：未来数据的平均分类误差率的期望值，可以通过对大规模相互独立的测试数据集的误分类率进行计算而近似得到，而不是用在模型训练过程中的。



* **K 折交叉验证（K-fold CV）**：

  把训练集分成 K 份，对每个$k\in \{1,...,K\}$都训练除了第 k 份之外的所有数据，然后在第 k 份上进行验证，就这样轮流进行，然后对所有份上的误差率计算均值，用这个作为测试误差的近似值

  > K = N，（N为样本数量）-》 留一法交叉验证（leave-one out cross validation，LOOCV）



> All models are wrong, but some models are useful. — George Box 











## C2. Probability

> Probability theory is nothing but common sense reduced to calculation. — Pierre Laplace, 1812



### 2.1 概率论

#### 2.1.1 离散随机变量

> 这个随机变量可以从任意的一个有限元素集合或者无限但可列的集合 *X* 中取值
>
> $p(x)$：概率质量函数（probability mass function，pmf）



#### 2.1.2 基本规则

* 结合概率：$p(A ∨ B) = p(A) + p(B) − p(A ∧ B)$

  > 当A，B互斥时，$p(A ∧ B) =0 $

  

* 联合概率：$p(A,B) = p(A ∧ B) = p(A\|B)p(B)$ （乘法规则）

* 对于联合概率分布$p(A,B)$，边缘分布定义为

  > $p(A)=\sum_b p(A,B) =\sum_b p(A\|B=b)p(B=b)$



* 链式规则

  > $p(X_{1:D})=p(X_1)p(X_2\|X_1)p(X_3\|X_2,X_1)p(X_4\|X_1,X_2,X_3) ... p(X_D\|X_{1:D-1})$



* 条件概率

  > $p(A\|B) = p(A,B)/p(B), \ p(B)>0 $ 



#### 2.1.3 贝叶斯定理

结合条件概率的定义以及乘法规则和加法规则，可以推出贝叶斯规则（Bayes rule）

> $p(X=x\|Y=y) = \frac{p(X=x,Y=y) }{p(Y=y) } = \frac{p(X=x)p(Y=y\|X=x)}{\sum_{x'}p(X=x')p(Y=y\|X=x') }$



* **生成分类器(Generative classifiers)**：使用类条件概率密度和类先验概率密度来确定如何生成数据

  > $p(y=c\|x,\theta)=\frac{p(y=c\|\theta)p(x\|y=c,\theta)}{\sum_{c'}p(y=c'\|\theta)p(x\|y= c',\theta)}$



#### 2.1.4 独立分布和无条件独立分布

* **无条件独立（unconditional independent）or 边缘独立（marginally independent）**

  > 记作 $X \bot Y$
  >
  > $X \bot Y \iff p(X,Y)=p(X)p(Y) $

  

一般来说，如果联合分布可以写成边缘的积的形式，就可以说一系列变量之间相互独立（mutually independent）

如果X 和 Y 对于给定的 Z 来说有条件独立，则可以将条件联合分布写成条件边缘的积的形式，即

> $X \bot Y \|Z \iff p(X,Y\|Z)=p(X\|Z)p(Y\|Z) $



**定理 2.2.1**

> $X\bot Y\| Z $则意味着存在着函数 g 和 h ，对全部的x,y,z，在 $p(z)>0$ 的情况下满足下列关系：
>
> $p(x,y\|z)=g(x,z)h(y,z)$



#### 2.1.5 连续随机变量

*  X 的**累计分布函数**（cumulative distribution function，缩写为 cdf）

  > $F(q) \overset\triangle{=} p(X\le q)$
  >
  > $p(a<X\le b) =F(b)-F(a)$

累积分布函数（cdf）是一个单调递增函数



* **概率密度函数**（probability density function，缩写为 pdf）

  > $f(x)=\frac{d}{dx}F(x)$
  >
  > $P(a<X\le b)= \int_a^b f(x)dx$
  >
  > 当微分区间区域无穷小时，得到以下形式：
  >
  > $P(x<X\le x+dx)\approx  p(x)dx$



pdf 要满足$p(x)\ge 0$，但对于任意的 x，$p(x)\ge 1$也有可能，只要密度函数最终积分应该等于1即可



#### 2.1.6 分位数

由于累积分布函数（cdf） F 是一个单调递增函数，那就有个反函数，记作$F^-$。如果 F 是 X 的累积分布函数（cdf），那么$F^{-1}(\alpha)$就是满足概率$P(X\le x_\alpha )=\alpha $的值；这也叫做 F 的 $\alpha$分位数（quantile）。

利用这个累积分布函数（cdf）的反函数还可以计算尾部概率（tail area probability），例子：高斯分布的 ” $3 \sigma$ 原则“ 



#### 2.1.7 均值和方差 （Mean & Variance）

* **均值**：期望值，记作$\mu$

> 离散随机变量： $\mathrm{E}[X] \overset\triangle{=} \sum_{x\in X}x p(x)$
>
> 连续随机变量： $\mathrm{E}[X] \overset\triangle{=} \int_{X}xp(x)dx$
>
> 注意：如果这个积分是无穷的，则均值不能定义



* **方差**：表征的是分布的“分散程度（spread）”，记作$\sigma^2$

$$
\begin{aligned}
var[X] * & =E[(X-\mu)^2]=\int(x-\mu)^2p(x)dx      &\text{           (2.24)}\\
& =  \int x^2p(x)dx  +\mu^2 \int p(x)dx-2\mu\int xp(x)dx=E[X^2]-\mu^2         &    \text{           (2.25)}\\
\end{aligned}
$$

则有，$E[X^2]= \mu^2+\sigma^2 $



* **标准差（standard deviation）**

> $std[X]\overset\triangle{=} \sqrt {var[X]}$
>
> 标准差和 X 单位相同



### 2.2 常见的离散分布

#### 2.2.1 二项分布 & 伯努利分布

* **二项分布**

  > $X\in \{0,...,n\}$
  >
  > $X\sim Bin(n,\theta)$
  >
  > 概率质量函数 pmf ：$Bin(k\|n,\theta)\overset\triangle{=} \binom{n}{k} \theta ^k  (1- \theta)^{n-k}$
  >
  > 其中，二项式系数 $ \binom{n}{k} \overset\triangle{=} \frac{n!}{(n-k)!k!}$

  均值 = $n\theta$，方差 = $n\theta(1- \theta)$



* **伯努利分布 or 0-1分布**

  > $X\in \{0,1 \}$
  >
  > $X\sim Ber(\theta)$
  >
  > 概率质量函数 pmf ：
  >
  > $Ber(x\|\theta)=\theta^{\prod (x=1)}(1-\theta)^{\prod (x=0)}$  or  
  >
  > $$Ber(x\|\theta)=\begin{cases} \theta &\text{ if x =1} \\
  > 1-\theta &\text{ if x =0} \end{cases}
  > $$ 

伯努利分布是二项分布中 n=1 的特例



#### 2.2.2 多项式（multinomial）分布和多重伯努利（multinoulli）分布

二项分布可以用于抛硬币这种情况的建模。要对有 K 个可能结果的事件进行建模，就要用到多项分布（multinomial distribution）。这个定义如下：设$x=(x_1,...,x_K)$ 是一个随机向量，其中的$x_j$是第 j 面出现的次数个数。

则 x 的概率质量函数 pmf 为
$$
Mu(x|n,\theta)\overset\triangle{=} {n}{x_1,..,x_K}\prod^K_{j=1}\theta^{x_j}_j
$$
其中的 $\theta_j$ 是第 j 面出现的概率，多项式系数 $\binom {n}{x_1,...,x_K} \overset\triangle{=} \frac{n!}{x_1!x_2!...x_K!}$ 表示将一个规模为$n=\sum^K_{k=1} x_k$的集合划分成规模从大小为${x_1、x_2\cdots x_k}$的 k 个子集的方案数

若 $n = 1$，则 x 变成 one-hot 编码，如 x = (1, 0, 0), x = (0, 1, 0), x = (0, 0, 1)

$Mu(x\|1,\theta)=\prod^K_{j=1 }\theta_j ^{\prod(x_j=1)}$

使用新的记号表示

$Cat(x\|\theta)\overset\triangle{=} Mu(x\|1,\theta)$

即，若 $x\sim Cat(\theta)$，则$p(x=j\|\theta)=\theta_j$



#### 2.2.3 泊松分布（Poisson Distribution）

如果一个离散随机变量$X\in \{0,1,2,...\}$服从泊松分布，即$X\sim  Poi(\lambda)$，其参数$\lambda >0$，其概率质量函数 pmf 为：

$$
Poi(x|\lambda )=e^{-\lambda}\frac{\lambda ^x}{x!}
$$
第一项是标准化常数（normalization constant），使用来保证概率密度函数的总和积分到一起是1 （联想到泰勒展开）

在二项分布的伯努利试验中，如果试验次数 n 很大，二项分布的概率 np 很小，且乘积$\lambda = np$ 比较适中，则事件出现的次数的概率可以用泊松分布来逼近

#### 2.2.4 经验分布

对于某个数据集$D =\{x_1,...,x_N \}$，就可以定义一个经验分布，也可以叫做经验测度（empirical measure），形式如下所示：

$$
p_{emp}(A)\overset\triangle{=}\frac 1 N \sum^N_{i=1}\delta _{x_i}(A)
$$
其中的$\delta_x(A)$是狄拉克测度（Dirac measure），定义为：
$$
\delta_x(A)= \begin{cases} 0 \text{    if }x \notin A \\
1\text{    if }x \in A 
\end{cases}
\
$$
一般来说可以给每个样本关联一个权重
$$
p(x)=\sum^N_{i=1}w_i\delta_{x_i}(x)
$$
其中, $0\le w_i \le 1$，$\sum^N_{i=1}w_i=1$



### 2.3 常见的连续分布

#### 2.3.1 高斯分布 or 正态分布

$$
N(x|\mu,\sigma^2) \overset\triangle{=} \frac {1}{\sqrt{2\pi \sigma^2}} e^ {-\frac{1}{2 \sigma^2}(x-\mu)^2} \\
\mu为均值/期望值，\sigma^2 = var[x]为方差，\sqrt{2\pi \sigma^2}是归一化常数，用于确保整个密度函数的积分是1
$$

* X服从高斯分布，即$X \sim  N(\mu,\sigma^2) $，表示$p(X=x)=N(x\|\mu,\sigma^2)$

* 标准正态分布：$X \sim N(0, 1)$

* 高斯分布的精准度：$\lambda =1/\sigma^2$

​		精确度高的意思也就是方差低，而整个分布很窄，对称分布在均值为中心的区域。

* 高斯分布的累积分布函数(cdf)：
  $$
  \phi(x;\mu , \sigma^2)\overset\triangle{=} \int^x_{-\infty}N(z|\mu,\sigma^2)dz 
  $$
  以误差函数 (error function, erf) 的形式来计算：
  $$
  \phi(x;\mu , \sigma^)\overset\triangle{=} \frac 1 2[1+erf(z/\sqrt2)] \\
  其中，z = (x-\mu) / \sigma，误差函数为\ erf(x)\overset\triangle{=} \frac{2}{\sqrt\pi}\int^x_0e^{-t^2}dt
  $$



#### 2.3.2 退化概率分布函数(Degenerate pdf)

如果让方差趋近于零，即$\sigma^2 \rightarrow 0$，那么高斯分布就变成高度为无穷大而峰值宽度无穷小的形状了，中心当然还是在$\mu$位置
$$
\lim_{\sigma^2\rightarrow 0} N(x| \mu,\sigma^2) =\delta (x-\mu)
$$
分布函数 $\delta$ 叫做狄拉克函数(Dirac delta function)。其定义如下
$$
\delta(x)=\begin{cases}\infty \text{   if   } x=0\\ 
0 \text{   if   } x\ne 0
\end{cases}
$$
并且有 $\int_{-\infty} ^\infty \delta(x)dx=1$

狄拉克函数具有筛选特性，可以从一系列求和或者积分当中筛选了单一项目：
$$
\int_{-\infty} ^\infty f(x) \delta(x-\mu )dx=f(\mu )
$$
只有当$x-\mu =0$的时候这个积分才是非零的



#### 2.3.3 学生分布（T分布）

高斯分布对异常值很敏感,因为从分布中心往外的对数概率衰减速度和距离成平方关系，T分布则表现得更为健壮，其概率密度函数如下：
$$
T(x|\mu,\sigma^2,v)\propto [1+\frac 1v (\frac{x-\mu}{\sigma})^2 ]^ {-(\frac {v+1}{2})}
$$


其中，$\mu$是均值, $\sigma^2>0$是范围参数(scale parameter), $v>0$称为自由度( degrees of freedom)

> $mean=\mu, mode=\mu,var=\frac{v\sigma^2}{v-2}$

这个模型中，当自由度大于2 （$v>2$）的时候方差才有意义，自由度大于1 （$v>1$）均值才有意义

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051721161.png)

* T 分布的稳定性如上图所示,左侧用的是没有异常值的高斯分布和T 分布,右侧是加入了异常值的.很明显这个异常值对于高斯分布来说干扰很大,而 T 分布则几乎看不出来有影响.因为 T 分布比高斯分布更重尾，至少对于小自由度 v 的时候是这样的。

* 当自由度 v = 1时， T 分布就成了柯西分布或者洛伦兹分布。注意：此时重尾会导致定义均值的积分不收敛。
* 要确保有限范围的方差, 就需要自由度 v > 2.一般常用的是自由度 v = 4, 在一系列问题中的性能表现也都不错. 如果自由度远超过5，即 v >> 5，T 分布就很快近似到高斯分布了,也就失去了健壮性。



#### 2.3.4 拉普拉斯分布 (Laplace distribution)

* 概率密度函数 pdf 
  $$
  Lap(x|\mu,b)\overset\triangle{=}\frac1{2b}\exp(-\frac{|x-\mu|}{b})
  $$
  其中，$\mu$是位置参数，b>0 是缩放参数

  > $mean=\mu, mode=\mu,var=2b^2$



#### 2.3.5 $\gamma$ 分布

* 概率密度函数
  $$
  Ga(T|shape=a, rate=b)\overset\triangle{=}  \frac{b^a}{\Gamma(a) }T^{a-1}e^{-Tb}
  $$
  其中，a>0 是形状参数，b>0是频率参数，$\Gamma(a)$ 是一个 $\gamma$ 函数，定义如下

  > $\Gamma(x) \overset\triangle{=} \int_0^{\infty} u^{x-1}e^{-u}du$
  >
  > $mean=\frac{a}{b}, mode=\frac{a-1}{b},var=\frac{a}{b^2}$



以下分布是$\gamma$分布的特例

* 指数分布 （Exponential distribution）

  > $Expon(x\|\lambda)\overset\triangle{=} Ga(x\|1,\lambda)$
  >
  > 其中的$\lambda$是频率参数

  这个分布描述的是泊松过程中事件之间的时间间隔.例如,一个过程可能有很多一系列事件按照某个固定的平均频率 $\lambda$ 连续独立发生.

  

* 爱尔朗分布（Erlang Distribution）

  > $Erlang(x\|\lambda) = Ga(x\|2, \lambda)$
  >
  > $\lambda$是频率参数

  爱尔朗分布是一个形状参数 a 是整数的$\gamma$分布，一般会设置 a=2

  

* 卡方分布 （Chi-squared distribution）

  > $\chi ^2 (x\|ν) \overset\triangle{=}Ga(x\|\fracν2,\frac12 )$

​		这个分布是高斯分布随机变量的平方和的分布。更确切地说，如果有一个高斯分布$Z_i \sim  N(0, 1)$，那么其平方和$S=\sum_{i=1}^vZ_i^2$则服从卡方分布$S \sim  \chi_v^2$



若一个随机变量服从$\gamma$分布：$X \sim  Ga(a,b)$ 那么这个随机变量的倒数就服从一个逆$\gamma$分布 (inverse gamma)，即$\frac 1X \sim  IG(a,b)$

> $IG(x\|shape =a,scale =b)\overset\triangle{=} \frac{b^a}{\Gamma  (a)}x^{-(a+1)} e^{-\frac b x}$
>
> $ mean= \frac{b}{a-1},mode=\frac{b}{a+1},var=\frac{b^2}{(a-1)^2(a-2)}$

注意：这个均值只在 a>1 的情况下才存在,而方差仅在 a>2 的时候存在。



#### 2.3.6 $\beta$ 分布

$\beta$分布支持区间 [0,1]，定义如下：

> $Beta(x\|a,b)=\frac{1}{B(a,b)}x^{a-1 }(1-x)^{b-1}$
>
> 其中，$B(a,b)$是一个$\beta$函数，定义为$B(a,b)\overset\triangle{=}\frac{\Gamma (a)\Gamma (b)}{\Gamma (a+b)}$
>
> $ mean= \frac{a}{a+b},mode=\frac{a-1}{a+b-2},var=\frac{ab}{(a+b)^2(a+b+1)}$

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051721658.png)

* 为了保证$B(a,b)$存在，需要 a 和 b 都大于零来确保整个分布可以积分
* 若 a=b=1, 得到的就是均匀分布
* 若 a 和 b 都小于1，则得到的就是一个双峰分布，两个峰值在0和1位置上
* 如果 a 和 b 都大于1，则得到的就是单峰分布



#### 2.3.7 帕累托分布 (Pareto distribution)

帕累托分布是用来对具有长尾或者重尾特点的变量进行建模的 （齐夫定律、二八定律）

概率密度函数 pdf 为

> $Pareto(x\|k,m)=km^kx^{-(k+1)}\prod(x\geq m)$
>
> $mean=\frac{km}{k-1} \text{   if }k>1,mode=m, var =\frac{m^2k}{(k-1)^2(k-2)} \text{   if }k>2$

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051722140.png)



### 2.4 联合概率分布

#### 2.4.1 协方差和相关系数

* **协方差 (covariance)** 是用来衡量两组变量之间 (线性) 相关的程度的

  > $cov[X,Y]\overset\triangle{=} E[(X-E[X])(Y-E[Y])] =E[XY]-E[X]E[Y] $

  若 x 是一个 d 维度的随机向量，则其协方差矩阵 (covariance matrix) 的定义如下
  $$
  \begin{aligned}
  cov[x] &= E[(x-E[x])(x-E[x])^T] \text{    }\\
  &=  \begin{pmatrix}
          var[X_1] & cov[X_1,X_2] &...& cov[X_1,X_d]  \\
          cov[X_2,X_1]  & var[X_2]  &...&cov[X_2,X_d]  \\
          ...&...&...&...\\
         cov[X_d,X_1]  & cov[X_d,X_2] &...&var[X_d] \\
          \end{pmatrix} \text{     }\\
  \end{aligned}
  $$
  该协方差矩阵为对称正定矩阵

  协方差可以从0到$\infty $之间取值，有时候为了使用方便，会将对其进行正规化



* **相关系数 (correlation coefficient)**

  > $corr[X,Y]\overset\triangle{=} \frac{cov[X,Y]}{\sqrt{var[X]var[Y]}}$

  相关矩阵为
  $$
  R= \begin{pmatrix}
          cov[X_1,X_1] & cov[X_1,X_2] &...& cov[X_1,X_d]  \\
          ...&...&...&...\\
         cov[X_d,X_1]  & cov[X_d,X_2] &...&var[X_d] \\
          \end{pmatrix}
     
  $$
  

  * 相关系数的取值范围为 [-1,1]，在一个相关矩阵中，每一个对角线项值都是1，其他的值都是在 [-1,1] 这个区间内

  * 当且仅当有参数 a 和 b 满足 $Y = aX + b$ 的时候,才有 $corr [X, Y ] = 1 $，即 $X$ 和 $Y$ 之间为线性关系

  * 相互独立可以推出不相关

    > $p(X, Y) = p(X)p(Y) \Rightarrow cov[X,Y]=0, \ corr[X, Y] = 0$

    但不相关不一定能推出相互独立



#### 2.4.2 多元高斯分布

在 D 维度上的多元正态分布(multivariate normal，MVN)的定义如下

> $N(x\|\mu,\Sigma)\overset\triangle{=} \frac{1}{(2\pi )^{\frac D2} \|\Sigma\|^{\frac12}}\exp [-\frac12 (x-\mu)^T\Sigma^{-1}(x-\mu) ]$

* $\mu = E [x] \in R^D$ 是均值向量
* $\Sigma= cov [x]$ 是一个$ D\times D$的协方差矩阵
* $\Lambda =\Sigma^{-1 }$ 是协方差矩阵的逆矩阵
* $(2\pi )^{\frac D2}\|\Sigma\|^{\frac12}$ 是归一化常数，为了保证概率密度函数的积分等于1



#### 2.4.3 多元学生 T 分布

相比多元正态分布 MVN，多元学生T 分布更加健壮，其概率密度函数为：
$$
\begin{aligned}
\Gamma (x|\mu,\Sigma,v)&=\frac{\Gamma (v/2+D/2)}{\Gamma (v/2+D/2)}  \frac{|\Sigma|^{-1/2}}{v^{D/2}\pi^{D/2}}\times [1+\frac1v(x-\mu )^T\Sigma^{-1}(x-\mu)]^{-(\frac{v+D}{2})}
&\text{   }\\
&=\frac{\Gamma (v/2+D/2)}{\Gamma (v/2+D/2)} |\pi V|^{-1/2}\times [1+(x-\mu)^T\Sigma^{-1}(x-\mu)]^{-(\frac{v+D}{2})}
 &\text{  }\\
\end{aligned}
$$

* $\Sigma$ 为范围矩阵 (scale matrix)，不是协方差矩阵
* $V=v\Sigma$
* T 分布比高斯分布有更重的尾部，并且参数 $v$ 越小，越重尾
* 当 $v\rightarrow \infty$ 时，T 分布趋向于高斯分布
* $mean=\mu, mode=\mu,  Cov=\frac{v}{v-2}\Sigma$



#### 2.4.4 狄利克雷分布

$\beta$分布扩展到多元 $\Rightarrow$ 狄利克雷分布（支持概率单纯形），定义如下

>$S_K={x:0 \le x_k \le 1, \sum ^K_{k=1}x_k=1}$

 概率密度函数为

> $Dir(x\|\alpha)\overset\triangle{=} \frac{1}{B(\alpha)} \prod^K_{k=1} x_k^{\alpha_k -1}\prod(x\in S_K)$

其中，$B(\alpha_1,...,\alpha_K)$是将$\beta$函数在 K 个变量上的自然推广，其定义为

> $B(\alpha)\overset\triangle{=} \frac{\prod^K_{k=1}\Gamma (\alpha_k)}{\Gamma (\alpha_0)}$
>
> $\alpha_0\overset\triangle{=} \sum^K_{k=1}\alpha_k$



狄利克雷分布的属性如下

> $ E[x_k]=\frac{\alpha_k}{\alpha_0}, \ mode[x_k]=\frac{\alpha_k-1}{\alpha_0-K}, \ var[x_k]=\frac{\alpha_k(\alpha_0-\alpha_k)}{\alpha_0^2(\alpha_0+1)}$



通常使用对称的狄利克雷分布,$\alpha_k=\alpha/K$. 这样则有方差$var[x_k]=\frac{K-1}{K^2(\alpha+1)}$. 这样增大$\alpha$就能降低方差,提高了模型精度



### 2.5 随机变量变换

#### 2.5.1 线性变换

假设 $y = f(x) = Ax + b$，则有

> $E[y] = E[Ax + b] = A \mu + b$
>
> $cov[y]=cov[Ax+b]=A\Sigma A^T$
>
> 其中，$\Sigma =cov[x]$

若$f(x)=a^Tx+b $，则对应地

> $E[a^Tx+b]=a^T\mu+b$
>
> $var[y]=var[a^Tx+b]=a\Sigma a^T$



#### 2.5.2 通用变换

若 X 是一个离散随机变量，且 $f(x)=y$，则 $y$ 的概率质量函数 pmf 为

> $p_y(y)=\sum_{x:f(x)=y}p_x(x)$



若 X 是连续的随机变量，对应地

> $P_y(y)\overset\triangle{=}P(Y\le y)=P(f(X)\le y)=P(X\in\{x\|f(x)\le y\})$
>
> $P_y(y)$ 为累积分布函数
>
> 在$f()$单调的情况下可以写作
>
> $P_y(y)\overset\triangle{=}P(Y\le y)=P(X\le f^{-1}(y))=P_x(f^{-1}(y))$

对累积分布函数求导：

> $p_y(y)\overset\triangle{=} \frac{d}{dy}P_y(y)=\frac{d}{dy}P_x(f^{-1}(y))=\frac{dx}{dy}\frac{d}{dx}P_x(x)=\frac{dx}{dy}p_x(x)$

 

因此，通用表达式为 $$p_y(y)=p_x(x)\|\frac{dx}{dy}\|$$ (变量转换公式)



#### 2.5.3 变量的多重变化

前面的结果可以推到多元分布上.设$f$是一个函数,从$R^n$映射到$R^n$, 设$y=f(x)$. 那么就有这个函数的雅可比矩阵 J(Jacobian matrix):

$$
J_{x\rightarrow y } * = \frac{\partial(y_1,...,y_n)}{\partial(x_1,...,x_n)}\overset\triangle{=}
\begin{pmatrix}
        \frac{\partial y_1}{\partial x_1} & ...& \frac{\partial y_1}{\partial x_n}  \\
        ...&...&...\\
       \frac{\partial y_n}{\partial x_1}   &...&\frac{\partial y_n}{\partial x_n} \\
        \end{pmatrix} 
$$




矩阵 J 的行列式\|det J\|表示的是在运行函数 f 的时候一个单位的超立方体的体积变化.
如果 f 是一个可逆映射(invertible mapping),就可以用逆映射$y\rightarrow x$的雅可比矩阵(Jacobian matrix) 来定义变换后随机变量的概率密度函数(pdf)

$p_y(y)=p_x(x)\|det(\frac{\partial x}{\partial y})\|=p_x(x)\|detJ_{y\rightarrow x}$


例子：假如要把一个概率密度函数从笛卡尔坐标系(Cartesian coordinates)的$x=(x_1,x_2)$ 转换到一个极坐标系(polar coordinates)$y=(r,\theta )$, 其中有对应关系:$x_1=r \cos \theta,x_3=r \sin \theta$.这样则有雅可比矩阵如下:

$$
J_{y\rightarrow x }=
\begin{pmatrix}
        \frac{\partial x_1}{\partial r}  &\frac{\partial x_1}{\partial \theta}  \\
       \frac{\partial x_2}{\partial r} &\frac{\partial x_2}{\partial \theta} \\
        \end{pmatrix} =
\begin{pmatrix}
        \cos \theta   & -r \sin \theta \\
        \sin \theta   &   r\cos \theta\\
        \end{pmatrix} 
$$


矩阵 J 的行列式为:

$\|det J\|=\|r\cos^2\theta+r\sin^2\theta\|=\|r\|$

因此:

$p_y(y)=p_x(x)\|det J\|$

$p_{r,\theta}(r,\theta)=p_{x_1,x_2}(x_1,x_2)r=p_{x_1,x_2}(r\cos\theta,r\sin\theta)r$



#### 2.5.4 中心极限定理

假设有独立同分布的 N 个随机变量，且不一定为正态分布，每个变量的均值和方差均为 $\mu$ 和 $\sigma ^2$。设 $S_N =\sum^N_{i=1 }X_i$ 是所有随机变量的和，当 N 增大时，$S_N$ 的分布会接近正态分布：

> $p(S_N=s)=\frac{1}{\sqrt{2\pi N\sigma^2}}\exp(-\frac{(s-N\mu)^2}{2N\sigma^2})$

令 $$ Z_N \overset\triangle{=} \frac{S_N-N_{\mu}}{\sigma\sqrt N} = \frac{\bar X-\mu}{\sigma/\sqrt N} $$，则 $Z_N$ 的分布会收敛到标准正态分布 $N(0, 1)$，其中样本均值为：$\bar X=\frac 1 N \sum^N_{i=1}x_i$





### 2.6 蒙特卡洛近似方法

应用蒙特卡罗方法，可以对任意的随机变量的函数进行近似估计。先简单取一些样本，然后计算这些样本的函数的算术平均值。这个过程如下所示：

>  $E[f(X)]=\int f(x)p(x)dx\approx \frac1S\sum^S_{s=1}f(x_s)$

上式中 $x_s \sim  p(X)$。这就叫做蒙特卡罗积分(Monte Carlo integration)，相比数值积分的一个优势就是在蒙特卡罗积分中只在具有不可忽略概率的地方进行评估计算，而数值积分会对固定网格范围内的所有点的函数进行评估计算。

通过调整函数$f()$,就能对很多有用的变量进行估计,比如:

* $\bar x =\frac 1S \sum^S_{s=1}x_s\rightarrow E[X]$
* $\frac 1 S\sum^S_{s=1}(x_s-\bar x)^2\rightarrow var[X]$
* $\frac 1 S \# \{x_s \le c\}\rightarrow P(X\le c)$
* 中位数(median)$\{x_1,...,x_S\}\rightarrow median(X)$





### 2.7 信息论

#### 2.7.1 信息熵 （Entropy）

随机变量 X 服从分布 p，这个随机变量的熵 (entropy) 则表示为$H(X)$或者$H(p)$，这是对随机变量不确定性的一个衡量。对于一个有 K 个状态的离散随机变量来说，其信息熵定义如下:

> $H(X)\overset\triangle{=}-\sum^K_{k=1}p(X=k)\log_2p(X=k)$

通常都用2作为对数底数，此时单位为 bit，若使用自然底数 e 为底数，则单位为 nats



#### 2.7.2 KL散度

**KL 散度** (Kullback-Leibler divergence) or **相对熵** (relative entropy)，可以用来衡量p和q两个概率分布的差异性

> $KL(p\|\|q)\overset\triangle{=}\sum^K_{k=1}p_k\log\frac{p_k}{q_k}$，or
>
> $KL(p\|\|q)=\sum_kp_k\log p_k - \sum_kp_k\log q_k =-H(p)+H(p,q)$

其中，$H(p,q)$ 为交叉熵

> $H(p,q)\overset\triangle{=}-\sum_kp_k\log q_k$



#### 定理 2.8.1 信息不等式(Information inequality) 

> $KL(p\|\|q)\ge 0$ 当且仅当 $p=q$的时候, KL 散度为0

证明过程见 p58



#### 2.7.3 互信息量（Mutual information）

**互信息量**表示联合分布 $p(X, Y)$ 和因式分布 $p(X)p(Y)$的相关性

> $I(X;Y)\overset\triangle{=}KL(p(X,Y)\|\|p(X)p(Y))=\sum_x\sum_yp(x,y)\log\frac{p(x,y)}{p(x)p(y)}$

$I(X;Y)\ge0$ 的等号当且仅当 $p(X,Y)=p(X)p(Y)$ 的时候成立，即当两个变量相互独立时，它们的互信息量为 0 .

使用条件熵来表示的形式为

> $I(X;Y)=H(X)-H(X\|Y)=H(Y)-H(Y\|X)$

其中，$H(Y\|X)$为条件熵，定义如下

> $H(Y\|X)=\sum_xp(x)H(Y\|X=x)$



**点互信息量** (pointwise mutual information，PMI）衡量的是与偶发事件相比，两个事件之间的差异

> $PMI(x,y)\overset\triangle{=} \log\frac{p(x,y)}{p(x)p(y)}= \log\frac{p(x\|y)}{p(x)}= \log\frac{p(y\|x)}{p(y)}$

显然，X 和 Y 的互信息量 MI 就是点互信息量 PMI 的期望值，因此有，

> $PMI(x,y)= \log\frac{p(x\|y)}{p(x)}= \log\frac{p(y\|x)}{p(y)}$







## C3. Generative models for discrete data

### 3.1 贝叶斯概念学习 (Bayesian concept learning)

#### 3.1.1 似然率

强抽样假设：假设样本是从概念的扩展集中随机抽取出来的（概念的扩展集就是所有属于该概念的元素组成的集合）

在强抽样假设的基础上，从 h 中可替换地独立抽取 N 个样本的概率为

> $p(D\|h)=[\frac{1}{size(h)}]^N=[\frac{1}{\|h\|}]^N$

规模原则 (size principle) or 奥卡姆剃刀 (Occam’s razor)：优先选择与数据样本一致且假设最少或者最简单的模型

两条似然原则：

* 似然函数包含了所有从实验中获得的包含未知参数的证据
* 如果一个似然函数A与另一个似然函数B成比例，那么A和B包含关于未知参数θ的信息相同



#### 3.1.2 先验（Prior）

贝叶斯统计的重要特点在于，我们在建模前需要给出模型参数 $\theta$ 的**先验分布**，即得到任何数据，或者将任何数据对模型进行拟合之前，我们需要先给定模型参数服从的分布。

**注意：先验分布是关于模型参数的分布，而不是建模的对象本身，下文中将要介绍的后验分布，也是关于模型参数的分布**



#### 3.1.3 后验 (Posterior)

后验就是似然率乘以先验，再进行归一化处理，即为

> $p(h\|D)=\frac{p(D\|h)p(h)}{\sum_{\hat h\in H}p(D, \hat h)}=\frac{p(h)I(D\in h)/\|h\|^N}{\sum_{\hat h\in H}p(\hat h)I(D\in h)/\|h\|^N}$

当且仅当所有数据都包含于假设h的扩展中的时候，其中的$I(D\in h)=1$



* 通常来说，只要有足够数据，后验概率密度$p(h\|D)$就会在一个单独概念位置有最大峰值，即为了最大后验(MAP)

  > $p(h\|D)\rightarrow \delta_{\hat h^{MAP}}(h)$

  其中，$\hat h^{MAP}= \arg \max_h p(h\|D)$是后验众数，$\delta$ 是狄拉克测度，定义为

  $$\delta_x(A)=\begin{cases}1 &\text{if}&x\in A\\
  0 &\text{if}&x\notin A\end{cases}
  $$

最大后验 (MAP) 可以写作:

> $\hat  h^{MAP} = \arg \max_h p(D\|h)p(h)=\arg \max_h [\log p(D\|h)+\log p(h)]$

由于似然率项依赖于N的指数函数，而先验保持不变，所以随着数据越来越多，最大后验估计(MAP estimate)就收敛到最大似然估计，则有

> $\hat  h^{mle} \overset\triangle{=} \arg \max_h p(D\|h) =\arg \max_h \log p(D\|h)$

换言之，如果有足够的数据，使得数据特征盖过了先验，此时最大后验估计就会朝着最大似然估计收敛



#### 3.1.4 后验预测分布 (Posterior predictive distribution)

* 后验预测分布 or 贝叶斯模型平均值：每个独立假设给出的预测的加权平均值

  > $p(\hat x \in C\|D )=\sum_h p(y=1\|\hat x,h)p(h\|D)$

  后验概率分布是以最大后验估计为中心的 $\delta$ 分布（对预测密度的插值近似），此时对应的预测分布为

  > $p(\hat x\in C\|D)=\sum_hp(\hat x\|h)\delta_{\hat h}(h)=p(\hat x\|\hat h)$





### 3.2 $\beta$ 二项模型（beta-binomial model）

#### 3.2.1 似然率

设 X 服从伯努利分布,即 $X_i \sim  Ber(\theta)$，$X_i=1$表示人头，$X_i=0$表示背面，$\theta \in [0, 1] $是频率参数(人头出现的概率)。如果实验事件是独立同分布的，则似然率为:

> $p(D\|\theta) =\theta^{N_1}(1-\theta)^{N-0}$

上式中的$N_1 =\sum^N_{i=1} I(x_i = 1)$对应人头出现的次数，而$N_0 =\sum^N_{i=1} I(x_i = 0)$ 对应背面出现的次数。这两个计数叫做数据的充分统计(sufficient statistics)，关于D我们只需要知道这两个量，就能推导$\theta$。充分统计集合也可以设置为$N_1$和$N = N_0 + N_1$。

**规范表述**

> 若$p(\theta\|D) = p(\theta\|s(data))$，则就可以称$s(D)$是对数据D的一个充分统计。如果使用均匀分布作为先验，也就等价说 $p(D\|\theta) \propto p(s(D)\|\theta)$。如果我们有两个集合，有同样的充分统计，就会推出同样的参数值$\theta$.



假设固定的总实验次数$N = N_0 + N_1$的情况下，数据中包含了人头朝上的次数为$N_1$。这时候就有$N_1$服从二项分布，即$N_1 \sim  Bin(N, \theta)$，其概率质量函数 pmf 为

> $Bin(k\|n,\theta))\overset\triangle{=}\binom{n}{k}\theta^k(1-\theta)^{n-k}$

因为$\binom{n}{k}$是独立于$\theta$的一个常数，所以二项取样模型的似然率和伯努利模型的似然率是一样的，所以我们对$\theta$的推断都是一样的，无论是观察一系列计数$D(N_1,N)$或者是有序的一系列测试$D = \{x_1 , ... , x_N \}$。



#### 3.2.2 先验

需要一个定义在区间[0,1]上的先验，为了数学运算简便，可以让先验和似然率形式相同，也就是说对于参数为$\gamma _1,\gamma_2$的某个先验来说：

> $p(\theta)\propto \theta^{\gamma_1}(1-\theta)^{\gamma_2}$

后验为

> $p(\theta) \propto p(D\|\theta)p(\theta) = \theta^{ N_1} (1 − \theta) ^{N_0 }\theta ^{\gamma_1 }(1 − \theta) ^{\gamma_2 }= \theta ^{N_1 +\gamma_1} (1 − \theta) ^{N_0 +\gamma_2}$

此时先验和后验形式相同，即该先验是所对应似然率的共轭先验

* 使用伯努利分布的情况下，共轭先验就是$\beta$分布，即

  > $Beta(\theta\|a,b)\propto \theta^{a-1}(1-\theta)^{b-1} $

* 无信息先验：仅知道参数 $\theta \in [0, 1]$，则可以使用均匀分布



#### 3.2.3 后验

把二项分布的似然率和$\beta$分布的先验相乘，即可得到下面的后验

> $p(\theta\|D ) \propto Bin(N_1\|N_0+ N_1,\theta)Beta(\theta\|a,b) \propto Beta(\theta\|N_1+a,N_0+b)$

该后验是通过在经验计数基础上加上了先验超参数而得到的.因此将这些超参数称之为伪计数

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051723745.png)

* 在图3.6 (a) 中，使用的是弱先验Beta(2,2)，似然率函数为单峰，对应一个大取样规模；从图中可见后验和似然率基本相符合：这是因为数据规模盖过了先验.
* 图3.6 (b) 是使用了强先验Beta(5,2)来进行更新，也是一个单峰值似然率函数，可这时候很明显后验就是在先验和似然率函数之间的一个折中调和.

注意：按顺序对后验进行更新等价于单次批量更新

假设有两个数据集$D_a,D_b$，各自都有充分统计$N^a_1 , N^a_0$和$N^b_1 , N^b_0$。设$N_1= N^a_1+N^b_1,N_0= N^a_0+N^b_0$则是联合数据集的充分统计

在批量模式(batch mode)下则有：

> $p(\theta\|D_a,D_b)\propto Bin(N_1\|\theta,N_1+N_0)Beta(\theta\|a,b)\propto Beta(\theta\|N_1+a,N_0+b)$

在序列模式(sequential mode)则有:

> $$\begin{aligned}
> p(\theta\|D_a,D_b) &\propto p(D_b\|\theta)p(\theta\|D_a)&\\
>  &\propto Bin(N^b_1\|\theta,N^b_1+N^b_0)Beta(\theta\|N^a_1+a,N^a_0+b)&\\
>  &\propto Beta(\theta \|N^aa_1+N^b_1+a,N^a_0+N^b_0+b) &\\
> \end{aligned}$$



##### 后验的均值和众数

* 最大后验估计为：

  > $\hat\theta_{MAP}=\frac{a+N_1-1}{a+b+N-2}$

* 若使用均匀分布先验，那么最大后验估计(MAP)就会降低成为最大似然估计，即为硬币人头朝上的经验分数

  > $\hat\theta_{MLE}=\frac{N_1}{N}$

* **后验均值**：$$\bar\theta = \frac{a+N_1}{a+b+N}$$

  后验均值是先验均值和最大似然估计的凸组合，表示的就是在这两者之间进行折中，兼顾了先验的已有观点以及数据提供的信息

设$\alpha_0 = a + b$是先验中的等效样本容量，控制的是先验强度，然后令先验均值为$m_1=a/\alpha_0$。然后后验均值可以表示为:

> $E[]=\frac{\alpha_0 m_1+N_1}{N+\alpha_0} = \frac{\alpha_0}{N+\alpha_0}m_1+\frac{N}{N+\alpha_0}\frac{N_1}{N}=\lambda m_1+(1-\lambda)\hat\theta_{MLE}$

上式中的$\lambda=\frac{\alpha_0}{N+\alpha_0}$为先验和后验的等效样本容量的比值。所以先验越弱，$\lambda$越小，而后验均值就更接近最大似然估计



##### 后验的方差

$\beta$后验的方差为

> $var[\theta\|D]=\frac{(a+N_1)(b+N_0)}{(a+N_1+b+N_0)^2(a+N_1+b+N_0+1)}$

当 $N >> a, b$ 时，可以对上面式子进行近似简化

> $var[\theta\|D]\approx \frac{N_1 N_0}{NNN}=\frac{\bar\theta(1-\bar\theta)}{N}$

其中的 $\bar\theta$ 即为最大似然估计，然后能得到估计结果的"误差项"，也就是后验标准差：

> $\sigma =\sqrt{var[\theta\|D]}\approx \sqrt{ \frac{\bar\theta(1-\bar\theta)}{N}}$

显然，不确定性以$1/\sqrt N$的速度降低。要注意这里的不确定性，也就是方差,在$\bar\theta=0.5$的时候最大,在$\bar\theta$接近0或者1的时候最小。这意味着确定硬币是否有偏差要比确定硬币结果是否合理公平更容易



#### 3.2.4 后验预测分布

设预测一个硬币落地后人头朝上在未来单次实验中的概率服从后验分布$Beta(a,b)$.则有:

$$
\begin{aligned}
p(\bar x=1|D )& =\int^1_0 p(x=1|\theta)p(\theta|D)d\theta \\
&=\int^1_0\theta Beta(\theta|a,b)d\theta=E[\theta|D]=\frac{a}{a+b}\end{aligned}
$$
在这个案例中，后验预测分布的均值和后验均值参数插值是等价的：

>  $p(\bar x\|D)=Ber(\bar x\|E[\theta\|D])$



##### 过拟合与黑天鹅悖论

* 对小规模数据进行估计的时候会经常出现零计数问题或者稀疏数据问题

* 使用贝叶斯方法来推导一个对这个问题的解决方案。使用均匀先验,所以a=b=1。这样对后验均值插值就得到了拉普拉斯继承规则：

  > $p(\tilde x =1\|D)=\frac{N_1+1}{N_1+N_0+2}$

  上式中包含了一种实践中的常规做法，就是对经验计数加1,归一化，然后插值，这也叫做加一光滑 (add-one smoothing)。要注意对最大后验估计插值就不会有这种光滑效果，因为这时候模的形式 $\hat\theta=\frac{N_1+a-1}{N+a+b-2}$，如果a=b=1就成了最大似然估计了。



##### 预测未来多次实验

设有 M 次未来实验,要去预测其中的人头朝上的次数 x.这个概率则为:

$$
\begin{aligned}
p(x|D,M)&= \int_0^1 Bin(x|\theta,M)Beta(\theta|a,b)d\theta\\
&=\binom{M}{x}\frac{1}{B(a,b)}\int_0^1\theta^x(1-\theta)^{M-x}\theta^{a-1}(1-\theta)^{b-1}d\theta \\
\end{aligned}
$$


这个积分正好就是$Beta(a+x, M−x+b)$这个分布的归一化常数。因此：

> $\int^1_0\theta^x(1-\theta)^{M-x}\theta^{a-1}(1-\theta)^{b-1}d\theta = B(x+a,M-x+b)$

因此就能发现后验预测分布如下所示，是一个(复合)$\beta$-二项分布：

> $Bb(x\|a,b,M) * = {\begin{pmatrix}M\\x\end{pmatrix}} \frac{B(x+a,M-x+b)}{B(a,b)}$



这个分布的均值和方差如下所示：

> $E[x]=M\frac{a}{a+b},var[x]=\frac{Mab}{(a+b)^2}\frac{(a+b+M)}{a+b+1}$(3.35)

如果$M=1$,则$x\in \{0,1\}$，均值为：

> $E[x\|D]=p(x=1\|D)=\frac{a}{a+b}$



### 3.3  狄利克雷-多项式模型 (Dirichlet-multinomial model)

#### 3.3.1 似然率

假设观测了N次掷骰子，得到的点数集合为$D=\{x_1,...,x_N\}$，其中$x_i\in \{1,...,K\}$.假设这个数据是独立同分布的，则似然率为

> $p(D\|\theta)=\prod^K_{k=1}\theta^{N_k}_k$

其中，$N_k=\sum^K_{i=1}I(y_i=k)$是事件k出现的次数（这也是该模型的充分统计）



#### 3.3.2 先验

> $Dir(\theta\|\alpha )= \frac{1}{B(\alpha)}\prod^K_{}\theta_k^{\alpha_{k-1}}I(x\in S_K)$



#### 3.3.3 后验

后验 = 先验 $\times$ 似然率，也是一个狄利克雷分布：
$$
\begin{aligned}
p(\theta|D)& \propto p(D|\theta)p(\theta)\\
& \propto \prod^K_{k=1}\theta^{N_k}_k\theta_k^{\alpha_k -1} = \prod^K_{k=1}\theta_k^{\alpha_k+N_k-1} \\
&  =Dir(\theta|\alpha_1+N_1,...,\alpha_K+N_K)\\
\end{aligned}
$$
显然，该后验世通过将先验的超参数（伪计数）$\alpha _k$ 加到经验计数 $N_k$ 上获得的

**最大后验估计**(MAP estimate)

* 强化约束条件$\sum_k\theta_k=1$，可通过拉格朗日乘数来实现
* 受约束的目标函数：拉格朗日函数
* 对似然率取对数、加上对先验取对数，然后加上约束条件

> $l(\theta,\lambda) =\sum_kN_k\log\theta_k+\sum_k(\alpha_k-1)\log\theta_k+\lambda(1-\sum_k\theta_k)$

为了简化表达，定义一个$\hat N_k\overset\triangle{=} N_k + \alpha_k − 1$，取关于$\lambda$的导数即可得到初始约束：

> $\frac{\partial l}{\partial \lambda}= (1-\sum_k\theta_k)=0$ 

利用总和为1这个约束条件就可以解出来$\lambda$：

>$$
>\sum_k\hat N_k =\lambda\sum_k \theta_k \\
>N+\alpha_0-K=\lambda
>$$

其中，$\alpha_0\overset\triangle{=} \sum^K_{k=1}\alpha_k$等于先验中的样本规模，此时的最大后验估计为

> $\hat\theta_k = \frac{N_k+\alpha_k-1}{N+\alpha_0-K}$



若使用均匀分布作为先验，即$\alpha_k=1$，则可以解出最大似然估计

> $\hat\theta_k=N_k/N$

这正好是k面出现次数的经验分数



#### 3.3.4 后验预测分布

对一个单次多重伯努利实验，其后验预测分布如下：
$$
\begin{aligned}
p(X=j|D)&=\int p(X=j|\theta)p(\theta|D)d\theta\\
&=\int p(X=j|\theta_j)[\int p(\theta_{-j},\theta_j|D)d\theta_{-j}]d\theta_j \\
&=\int \theta_j p(\theta_j |D)d\theta =E[\theta_j|D]=\frac{\alpha_j+N_j}{\sum_k(\alpha_k+N_k)}=\frac{\alpha_j+N_j}{\alpha_0+N}\\
\end{aligned}
$$


注：$\theta_{-j}$是除了$\theta_j$之外的其他所有$\theta$的成员



### 3.4 朴素贝叶斯分类器 (Naive Bayes classiﬁers)



#### 3.4.1 模型拟合

##### 3.4.1.1 朴素贝叶斯分类器的最大似然估计

* 单数据情况下的概率为

  > $p(x_i,y_i\|\theta)=p(y_i\|\pi)\prod_jp(x_{ij}\|\theta_j)=\prod_c\pi_c^{I(y_i=c)}\prod_j\prod_cp(x_{ij}\|\theta_{jc})^{I(y_i=c)}  $

  对数似然率（log-likelihood）：

  > $\log p(D\|\theta) =\sum^C_{c=1}N_c\log\pi_c+\sum^D_{j=1}\sum^C_{c=1}\sum_{i:y_i=c}\log p(x_{ij}\|\theta_{jc})$

  

* 分类先验的最大似然估计为：

  > $\hat\pi_c =\frac{N_c}{N}$

  其中，$N_c\overset\triangle{=}\sum_iI(y_i=c)$ 是类c中的样本个数



* 对似然率的最大似然估计依赖于我们对特征所选的分布类型。简单起见，假设所有特征都是二值化的，这样使用伯努利分布，即

  > $x_j\|y=c\sim  Ber(\theta_{jc} )$

  这时候最大似然估计则为：

  > $\hat\theta_{jc}=\frac {N_{jc}}{N_c}$



##### 	3.4.1.2 使用贝叶斯方法的朴素贝叶斯 (Bayesian naive Bayes)

最大似然估计有个麻烦就是可能会过拟合，避免过拟合的简单解决方案就是使用贝叶斯方法。

因式化先验

> $p(\theta)=p(\pi)\prod^D_{j=1}\prod^C_{c=1}p(\theta_{jc})$

对于$\pi$使用狄利克雷先验$Dir(\alpha)$，对每个参数$\theta_{jc}$采用$\beta$分布$Beta(\beta_0,\beta_1)$。通常就设$\alpha=1,\beta=1$对应的是加一光滑或者拉普拉斯光滑。

结合前面的因式化似然率与因式化先验，即可得到下面的因式化后验

> $p(\theta\|D)=p(\pi\|D) \prod^D_{j=1}\prod^C_{c=1}p(\theta_{jc}\|D)$
>
> $p(\pi\|D)=Dir(N_1+\alpha_1,...,N_C+\alpha_C)$
>
> $p(\theta_{jc}\|D)=Beta((N_c-N_{jc})+\beta_0,N_{jc}+\beta_1)$



#### 3.4.2 使用模型做预测

计算目标：

> $p(y=c\|x,D)\propto p(y=c\|D)\prod^D_{j=1}p(x_j\|y=c\|D)$



* 使用积分排除掉未知参数

$$
\begin{aligned}
p(y=c|x,D)\propto & [\int Cat(y=c|\pi)p(\pi|D)d\pi]\\
&\prod^D_{j=1}[\int Ber(x_j|y=c,\theta_{jc})p(\theta_{jc}|D)]\\
\end{aligned}
$$

* 插入后验均值参数$\theta$来获得后验预测密度

$$
\begin{aligned}
p(y=c|x,D) &\propto  \bar\pi _C\prod^D_{j=1}(\bar\theta_{jc})^{I(x_j=1)} (1-\bar\theta_{jc})^{I(x_j=0)}  \\
\bar\theta_{jk} & =\frac{N_{jc}+\beta_1}{N_c+\beta_0+\beta_1} \\
\bar\pi_c & =\frac{N_c+\alpha_c}{N +\alpha_0} \\
\end{aligned}
$$

其中，$\alpha_0 = \sum _c \alpha_c$



如果我们通过单个点估计了后验,$p(\theta\|D)\approx \delta_{\hat\theta}(\theta) $，其中的$\hat\theta$可以使最大似然估计(MLE)或者最大后验估计，然后就可以通过对参数插值来得到后验预测密度了,生成的是一个虚拟一致规则：

> $p(y=c\|x,D)\propto \hat\pi_c\prod^D_{j=1}(\hat\theta_{jc})^{I(x_j=1}(1-\hat\theta_{jc})^{I(x_j=0)}$ 



#### 3.4.3 The log-sum-exp trick

* 应用贝叶斯规则的时候先取对数

$$
\begin{aligned}
\log p(y=c|X)&= b_c-\log {[} \sum^C_{c'=1}e^{b_{c'}}]\\
 b_c& \overset\triangle{=}\log p(x|y=c)+\log p(y=c) 
\end{aligned}
$$

* 计算对数项表达式

$$
\log{[}\sum_{c'}e^{b_{c'}}]  = \log[\sum_{c'}p(y=c',x)]=\log p(x)
$$

* 提取最大因子项，并简化

$$
\log(e + e^{−121} ) = \log e −120 (e^0 + e^{−1} ) = \log(e^0 + e^{−1} ) − 120 \\
\log\sum_ce^{b_c} =\log {[}(\sum_ce^{b_c-B})e^B]=[\log(\sum_ce^{b_c-B})]+B （一般得到的结果）\\
$$

​	最大公因式项$B=\max_cb_c$



#### 3.4.4 使用互信息量进行特征选择

为了降低开销，可以进行特征选择，剔除一些无关信息。最简单的信息选择方法是单独评价每个特征的相关性，选择最大的 K 个，K 是根据精确度和复杂度之间的权衡来选择的。（变量排序/过滤/筛选）

衡量相关性的一个手段就是利用互信息量

> 特征$X_j$和分类标签Y之间的互信息量：
>
> $I(X,Y) = \sum_{x_j} \sum_y p(x_j,y) \log \frac{p(x_j,y)}{p(x_j)p(y)} $



互信息量可以被理解为在观测特征 j 的值的时候标签分布上的信息熵降低，如果特征是二值化的，则可以用下面的公式来计算

> $I_j=\sum_c[\theta_{jc}\pi_c\log\frac{\theta_{jc}}{\theta_{j}}+ (1-\theta_{jc})\pi)c\log\frac{1-\theta_{jc}}{1-\theta_{j}}  ]$

其中，$\pi_c=p(y=c),\theta_{jc}=p(x_j=1\|y=c),\theta_j=p(x_j=1)=\sum_c\pi_c\theta_{jc}$





## C4. Gaussian models

### 4.1 基础知识

#### 4.1.1 多元正态分布

* 概率密度函数：

> $N(x\|\mu,\Sigma)\overset \triangle{=} \frac{1}{(2\pi)^{D/2}\|\Sigma \|^{1/2}}\exp[ -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)]$



* 马氏距离 (马哈拉诺比斯距离)

$$
\begin{aligned}
(x-\mu)^T\Sigma^{-1}(x-\mu)&=(x-\mu)^T(\sum^D_{i=1}\frac{1}{\lambda_i}u_iu_i^T)(x-\mu)\\
&= \sum^D_{i=1}\frac{1}{\lambda_i}(x-\mu)^Tu_iu_i^T(x-\mu)=\sum^D_{i=1}\frac{y_i^2}{\lambda_i}\\
\end{aligned}
$$

其中，$y_i\overset{}{=} u_i^T(x-\mu)$，二维椭圆方程为 $\frac{y_1^2}{\lambda_1}+\frac{y_2^2}{\lambda_2}=1$

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051728577.png)

* 高斯分布的概率密度的等值线沿着椭圆形，如上图所示。特征向量决定了椭圆的方向，特征值决定了椭圆的形态即宽窄比。
* 一般可以将马氏距离看作是对应着变换（平移$\mu$，旋转U）后坐标系中的欧氏距离



#### 4.1.2  多元正态分布的最大似然估计

#### 定理4.1.1(MVN的MLE)	

如果有N个独立同分布样本符合正态分布，即$x_i \sim  N(\mu,\Sigma)$,则对参数的最大似然估计为：
$\hat\mu_{mle}=\frac{1}{N}\sum^N_{i=1}x_i \overset \triangle{=} \bar x$
$\hat\Sigma_{mle}=\frac{1}{N}\sum^N_{i=1}(x_i-\bar x)(x_i-\bar x)^T=\frac{1}{N}(\sum^N_{i=1}x_ix_i^T)-\bar x\bar x^T$

也就是MLE就是经验均值和经验协方差。在单变量情况下结果为：
$\hat\mu =\frac{1}{N}\sum_ix_i=\bar x$
$\hat\sigma^2 =\frac{1}{N}\sum_i(x_i-x)^2=(\frac{1}{N}\sum_ix_i^2)-\bar x^2$





### 4.2 高斯判别分析 (Gaussian discriminant analysis，GDA)

**生成分类器**

> $p(y=c\|x,\theta)=\frac{p(y=c\|\theta)p(x\|y=c,\theta)}{\sum_{\dot c}p(y=\dot c\|\theta)p(x\|y=\dot  c,\theta)}$ （4.1）



多元正态分布的一个重要用途就是在生成分类器中定义类条件密度，即

> $$p(x\|y=c,\theta)=N(x\|\mu_c,\Sigma_c)$$

这样就得到了高斯判别分析（GDA）（生成分类器，而不是判别分类器）



从上述的等式 4.1 中可以推导出来下面的决策规则，对一个特征向量进行分类：

> $\hat y(x)= \underset{c}{\arg \max} [\log  p(y=c\|\pi)  +\log p(x\|\theta_c)]$

计算 x 属于每一个类条件密度的概率的时候，测量的距离是x到每个类别中心的马氏距离，这也是一种最近邻质心分类器 (nearest centroids classiﬁer)。



#### 4.2.1 二次判别分析(Quadratic discriminant analysis，QDA)

对类标签的后验如等式 4.1 所示，在加入高斯分布概率密度定义后，即可得到
$$
p(y=c|x,\theta)  =\frac{ \pi_c|2\pi\Sigma_c|^{-1/2} \exp [-1/2(x-\mu_c)^T\Sigma_c^{-1}(x-\mu_c)]   }{   \Sigma_{c'}\pi_{c'}|2\pi\Sigma_{c'}|^{-1/2} \exp [-1/2(x-\mu_{c'})^T\Sigma_{c'}^{-1}(x-\mu_{c'})]}  \ \text{(4.33) }
$$
对此进行阈值处理即可得到一个 x 的二次函数，即为二次判别分析（QDA）

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051728318.png)



#### 4.2.2 线性判别分析 (Linear discriminant analysis，LDA)

考虑一种特殊情况：协方差矩阵为各类共享，即 $\Sigma_c = \Sigma$，此时可以将等式4.33 简化为以下形式：
$$
\begin{aligned}
p(y=c|x,\theta)&\propto \pi_c\exp [\mu_c^T\Sigma^{-1}x-\frac12 x^T\Sigma^{-1}x - \frac12\mu_c^T\Sigma^{-1}\mu_c]\\
& = \exp [\mu_c^T\Sigma^{-1}x-\frac12 \mu_c^T\Sigma^{-1}\mu_c+\log\pi_c]\exp [-\frac12 x^T\Sigma^{-1}x]\\
\end{aligned}
$$
由于二次项 $x^T \Sigma ^{-1}x$ 独立于类别 $c$，因此可以抵消掉分子分母，如果定义了：

> $\begin{aligned}
> \gamma_c &= -\frac12\mu_c^T\Sigma^{-1}\mu_c+\log\pi_c \\
> \beta_c &= \Sigma^{-1}\mu_c\end{aligned}$

则有：

> $p(y=c\|x,\theta)=\frac{e^{\beta^T_c+\gamma_c}}{\Sigma_{c'}e^{\beta^T_{c'}+\gamma_{c'}}}=S(\eta)_c  \  \ \text{(4.38)}$

其中，$\eta =[\beta^T_1x+\gamma_1,...,\beta^T_Cx+\gamma_C]$，$S$函数为 $Softmax$ 函数，其定义如下：

> $S(\eta)= \frac{e^{\eta_c}}{\sum^C_{c'=1}e^{\eta_{c'}}}$



将每个 $\eta_c$ 除以一个常数 $T$（温度），当 $T \rightarrow 0$ 时，有:
$$
S(\eta/T)_c=\begin{cases} 1.0&\text{if } c = \arg\max_{c'}\eta_{c'}\\
0.0 &\text{otherwise}\end{cases} 
$$
换言之，,在低温情况下，分布总体基本都出现在最高概率的状态下；而在高温下，分布会均匀分布于所有状态，这个概念来自统计物理性,通常称为玻尔兹曼分布（和 Softmax 函数的形式相同）

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051728000.png)





#### 4.2.3  双类线性判别分析 (Two-class LDA)

先考虑二值化分类的情况，，此时后验为：
$$
\begin{aligned}
p(y=1|x,\theta)& =\frac{e^{\beta^T_1x+\gamma_1}}{e^{\beta^T_1x+\gamma_1}+e^{\beta^T_0x+\gamma_0}}    &\text{(4.44)}\\
& = \frac{1}{1+e^{(\beta_0-\beta_1))^Tx+(\gamma_0-\gamma_1)}} =sigm((\beta_1-\beta_0)^Tx+(\gamma_1-\gamma_0))  &\text{(4.45)}\\
\end{aligned}
$$
上式中的 $sigm(\eta)$即为 S 型函数（sigmoid function），因此有：
$$
\begin{aligned}
\gamma_1-\gamma_0 & = -\frac{1}{2}\mu^T_1\Sigma^{-1}\mu_1+\frac{1}{2}\mu^T_0\Sigma^{-1}\mu_0+\log(\pi_1/\pi_0) &\text{(4.46)}\\
& =  -\frac{1}{2}(\mu_1-\mu_0)^T\Sigma^{-1}(\mu_1+\mu_0) +\log(\pi_1/\pi_0) &\text{(4.47)}\\
\end{aligned}
$$
如果定义了：
$$
\begin{aligned}
w&=  \beta_1-\beta_0=\Sigma^{-1}(\mu_1-\mu_0)&\text{(4.48)}\\
x_0 & =  -\frac{1}{2}(\mu_1+\mu_0)-(\mu_1-\mu_0)\frac{\log(\pi_1/\pi_0) }{(\mu_1-\mu_0)^T\Sigma^{-1}(\mu_1-\mu_0)} &\text{(4.49)}\\
\end{aligned}
$$
则有 $w^Tx_0=-(\gamma_1-\gamma_0)$，因此：
$$
p(y=1|x,\theta) = sigm(w^T(x-x_0))
$$


最终的决策规则为：

* 将 $x$ 移动 $x_0$ ，然后投影到直线 $w$ 上，观察结果的正负号
* 若 $\Sigma = \sigma^2 I$ ，则 $w$ 就是 $\mu_1 - \mu_0 $ 的方向，对点进行分类需要根据其投影距离 $\mu_1$ 和 $\mu_0$ 哪个更近
* 若 $\pi_1 = \pi_0$，那么 $x_0 = \frac{1}{2} (\mu_0 + \mu_1)$，正好位于两个均值的中间位置；若$\pi_1> \pi_0$，则$x_0$更接近$\mu_0$；反之，若$\pi_1 < \pi_0$，则$x_0$更接近$\mu_1$（边界右移）

由此可见，类的先验$\pi_c$只是改变了决策阈值，而并没有改变总体的结合形态，$w$的大小决定了对数函数的陡峭程度,取决于均值相对于方差的平均分离程度。

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051728920.png)

在心理学和信号检测理论中,通常定义一个叫做敏感度指数 (sensitivity index,也称作 d-prime) 的量，表示信号和背景噪声的可区别程度:

> $d'\overset{*}{=} \frac{\mu_1-\mu_0}{\sigma}$

上式中的$\mu_1$是信号均值，$\mu_0$是噪音均值，而$\sigma$是噪音的标准差。如果敏感度指数很大,那么就意味着信号更容易从噪音中提取出来。





#### 4.2.4 对于判别分析的最大似然估计

拟合一个判别分析模型最简单的方法即为最大似然估计，对应的对数似然函数为：

> $\log p(D\|\theta) =[\sum^N_{i=1}\sum^C_{c=1}I(y_i=c)\log\pi_c] + \sum^C_{c=1}[\sum_{i:y_i=c}\log N(x\|\mu_c,\Sigma_c)]$

上式可以因式分解成一个含有$\pi$的项，以及对应每个$\mu_c, \Sigma_c$ 的 c 个项，可以分开对这些参数进行估计。

* 对于类先验，有 $\hat \pi_c = \frac{N_c} {N}$（和朴素贝叶斯分类器一样）

* 对于类条件密度，可以根据数据的类别标签来分开，对于每个高斯分布进行最大似然估计：

  > $\hat\mu_c=\frac{1}{N_c}\sum_{i:y_i=c}x_i, \ \hat\Sigma_c=\frac{1}{N_c}\sum_{i:y_i=c}(x_i-\hat\mu_c)(x_i-\hat\mu_c)^T$



#### 4.2.5 防止过拟合的策略

最大似然估计（MLE）

* 优势是速度和简洁
* 在高维数据的情况下，最大似然估计可能会发生过拟合，尤其是当 $N_c < D$ ，全协方差矩阵是奇异矩阵时，MLE方差很容易发生过拟合
* 即便在 $N_c > D$ 时，MLE可能也是存在问题的 (ill-conditioned)



预防和解决过拟合问题的策略：

* 假设类的特征是有条件独立的，对这些类使用对角协方差矩阵；（等价于使用朴素贝叶斯分类器）
* 使用一个全协方差矩阵，但强制使其对于所有的类都相同，即$\Sigma_c=\Sigma$。这称为参数绑定或者参数共享；（等价于线性判别分析(LDA)）
* 使用一个对角协方差矩阵，强迫共享；（对角协方差线性判别分析）
* 使用全协方差矩阵，但加入一个先验，然后整合，如果使用共轭先验就能以闭合形式完成这个过程；
* 拟合一个完整的或者对角协方差矩阵，使用最大后验估计（MAP）
* 将数据投影到更低维度的子空间，然后在子空间中拟合其高斯分布。														



#### 4.2.6 正交线性判别分析(Regularized LDA) *

​	略



#### 4.2.7 对角线性判别分析 (Diagonal LDA)

正交线性判别分析（RDA）有一种简单的替代方法：

>  绑定协方差矩阵，即线性判别分析中 $\Sigma_c = \Sigma$，然后对于每个类都使用一个对角协方差矩阵 （对角线性判别分析模型，等价于 $\lambda = 1$ 时候的正交线性判别分析 ）



对角线性判别分析对应的判别函数为：

> $\delta _c(x)=\log p(x,y=c\|\theta) =-\sum^D_{j=1}\frac{(x_j-\mu_{cj})^2}{2\sigma^2_j}+\log\pi_c$

通常设置 $\hat\mu_{cj}=\bar x_{cj},\hat\sigma^2_j=s^2_j$，其中 $s_j^2$ 是特征 $j$ 的汇集经验方差（跨类汇集）

> $s^2_j=\frac{\sum^C_{c=1}\sum_{i:y_i=c}(x_{ij}-\bar x_{cj})^2}{N-C}$

对于高维度数据，这个模型比LDA和RDA效果更好



#### 4.2.8 最近收缩质心分类器 (Nearest shrunken centroids classiﬁer) *

​	略



### 4.3 联合正态分布的推论 (Inference in jointly Gaussian distributions)

#### 4.3.1 结果声明

##### 定理 4.3.1（多元正态分布的边界和条件分布）

设$x=(x_1,x_2)$是联合正态分布，其参数如下：
$$
\mu=\begin{pmatrix}
        \mu_1\\
        \mu_2
        \end{pmatrix} ,
\Sigma=\begin{pmatrix}
        \Sigma_{11}&\Sigma_{12}\\
        \Sigma_{21}&\Sigma_{22}
        \end{pmatrix},
\Lambda=\Sigma^{-1}=\begin{pmatrix}
        \Lambda_{11}&\Lambda_{12}\\
        \Lambda_{21}&\Lambda_{22}
        \end{pmatrix}
\text{  (4.67)}
$$
边缘分布为
$$
\begin{aligned}
p(x_1)&=N(x_1|\mu_1,\Sigma_{11})\\
p(x_2)&=N(x_2|\mu_2,\Sigma_{22})
\end{aligned}
$$
后验条件分布为
$$
\begin{aligned}
p(x_1|x_2)&=N(x_1|\mu_{1|2},\Sigma_{1|2})\\
\mu_{1|2}&=\mu_1+\Sigma_{12}\Sigma^{-1}_{1|2}(x_2-\mu_2)\\
&=\mu_1-\Lambda_{12}\Lambda^{-1}_{1|2}(x_2-\mu_2)\\
&= \Sigma_{1|2}(\Lambda_{11}\mu_1-\Lambda_{12}(x_2-\mu_2))\\
\Sigma_{1|2}&=\Sigma_{11}-\Sigma_{12}\Sigma^{-1}_{22}\Sigma_{21}=\Lambda^{-1}_{11}
\end{aligned} 
$$
由此可见，边缘和条件分布本身也是正态分布

* 对于边缘分布，只需要提取出与 $x_1$ 或者 $x_2$ 对应的行和列
* 对于条件分布，条件均值正好是 $x_2$ 的一个线性函数，而条件协方差则是一个独立于 $x_2$ 的常数矩阵，上面给出了后验均值的三种不同的等价表达形式，以及后验协方差的两种不同的等价表达形式，每个表达式在不同情境下有各自的作用。



#### 4.3.2 Examples

##### 4.3.2.1 二维正态分布的边缘和条件分布

假设以一个二维正态分布为例，其协方差矩阵为：
$$
\Sigma =\begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2   \\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{pmatrix}
$$
边缘分布$p(x_1)$则是一个一维正态分布，将联合分布投影到$x_1$这条线上即可得到：
$$
p(x_1)=N(x_1|\mu_1,\sigma_1^2)p(x_1)=N(x_1|\mu_1,\sigma_1^2)
$$
假如观测$X_2=x_2$，则可以通过使用$X_2=x_2$这条线来对联合分布进行 "切片(slicing)" 得到条件分布$p(x_1\|x_2)$
$$
p(x_1|x_2)= N(x_1|\mu_1+ \frac{\rho\sigma_1\sigma_2}{\sigma_2^2}(x_2-\mu_2),\sigma_1^2-\frac{(\rho\sigma_1\sigma_2 )^2}{\sigma_2^2})
$$
若 $\sigma_1 = \sigma_2 = \sigma$，则有：
$$
p(x_1|x_2)=N(x_1|\mu_1+\rho(x_2-\mu_2),\sigma^2(1-\rho^2))
$$
![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051729081.png)

在上图所示的例子中，$\rho=0.8, \ \sigma_1 = \sigma_2 = 1, \ \mu = 0, \ x_2 = 1$。容易可见 $E[x_1 \| x_2 = 1] = 0.8$，$\rho = 0.8$ 意味着 $x_2$ 在均值的基础上加 1 ，则 $x_1$ 将会增加 0.8；另外我们还可以发现 $var[x_1 \| x_2 = 1] = 1 - 0.8^2 = 0.36$，其表达的意义为：由于通过观测$x_2$而对$x_1$有了非直接的了解,所以对$x_1$的不确定性就降低了；如果 $\rho = 0$，则可得到 $p(x_1\|x_2)=N(x_1\|\mu_1,\sigma_1^2)$，因为如果二者不相关，即互相独立的话，$x_2$ 将不会表达关于$x_1$的任何信息。



##### 4.3.2.2 无噪音数据插值 (Interpolating noise-free data)



##### 4.3.2.3 数据插补 (Data imputation)





#### 4.3.3 信息形式 (Information form)

设$x \sim  N(\mu,\Sigma)$，很明显$E[x]=\mu$就是均值向量，而$cov[x]=\Sigma$就是协方差矩阵，这些都叫做分布的矩参数。不过有时候可能使用规范参数或者自然参数更有用，其具体定义为：

> $\Lambda \overset{*}{=} \Sigma^{-1}, \ \xi \overset{*}{=}  \Sigma^{-1} \mu$

还可以转换回矩参数:

> $ \mu=\Lambda^{-1}\xi, \Sigma =\Lambda^{-1}$



使用规范参数，可以将多元正态分布写成信息形式（即指数组分布的形式），具体定义为：
$$
N_c(x|\xi,\Lambda)=(2\pi)^{-D/2}|\Lambda|^{\frac{1}{2}} \exp[-\frac{1}{2}(x^T\Lambda x+\xi^T\Lambda^{-1}\xi-2x^T\xi)]
$$
上式中使用了$N_c()$是为了和矩参数表达形式$N()$相区分



边缘分布和条件分布公式也都可以推导出信息形式：
$$
\begin{aligned}
p(x_2)&=N_c(x_2|\xi_2-\Lambda_{21}\Lambda_{11}^{-1}\xi_1,\Lambda_{22}-\Lambda_{21}\Lambda_{11}^{-1}\Lambda_{12}) \\
p(x_1|x_2)&=N_c(x_1|\xi_1-\Lambda_{12}x_2,\Lambda_{11})
\end{aligned}
$$
由上式可见，在矩参数形式下求边缘分布更容易，在信息形式下求条件分布更容易

信息形式记法的另一个好处是两个正态分布的相乘操作更为简单：

> $N_c(\xi_f,\lambda_f)N_c(\xi_g,\lambda_g)=N_c(\xi_f+\xi_g,\lambda_f+\lambda_g)$

而在矩参数形式下，相乘操作比较麻烦：

> $N(\mu_f,\sigma_f^2)N(\mu_g,\sigma_g^2)=N(\frac{\mu_f\sigma_g^2+\mu_g\sigma_f^2}{\sigma_f^2+\sigma_g^2},\frac{\sigma_f^2 \sigma_g^2}{\sigma_f^2+\sigma_g^2})$



#### 4.3.4 结论证明 *

​	略





### 4.4 线性高斯系统 (Linear Gaussian systems)

假设有两个变量 $x$ 和 $y$，设 $x \in R^{D_x}$ 是隐藏变量，而 $y \in R^{D_y}$ 是对 $x$ 的有噪声观察，假设有如下的先验和似然率：
$$
\begin{aligned}
p(x)&= N(x|\mu_x,\Sigma_x)   \\
p(y|x)&= N(y|Ax+b,\Sigma_y)   \\
\end{aligned}
$$
其中 $A$ 是一个 $D_x \times D_y$ 的矩阵，这就是一个线性高斯系统，可以表示为 $x \rightarrow y$，意思是 $x$ 生成了 $y$，本节主要解释如何逆转箭头方向，即根据 $y$ 来推测 $x$ 



#### 4.4.1 结论表述

##### 定理 4.4.1 线性高斯系统的贝叶斯规则

给定一个线性高斯系统，其形式如上述等式所示，则后验 $p(x \| y)$ 为：
$$
\begin{aligned}
p(x|y)&= N(x|\mu_{x|y},\Sigma_{x|y})   \\
\Sigma_{x|y}^{-1}&=\Sigma_x^{-1}+A^T\Sigma_y^{-1}A  \\
\mu_{x|y}&= \Sigma_{x|y}[A^T\Sigma_y^{-1}(y-b)+\Sigma_x^{-1}\mu_x] \\
\end{aligned}
$$


归一化常数 $p(y)$ 为：
$$
p(y)=N(y|A\mu_x+b,\Sigma_y+A\Sigma_x A^T)
$$


#### 4.4.2 Examples *

​	略



#### 4.4.3 结论证明 *

​	略





### 4.5 题外话: 威沙特分布(Wishart distribution) *

​	略





### 4.6 多元正态分布的参数推测

#### 4.6.1 $\mu$的后验分布

似然函数形式为：

> $p(D\|\mu)=N(\bar x\|\mu,\frac{1}{N}\Sigma)$

为了简化，我们使用共轭先验，这里的例子使用的是一个高斯分布。如果 $p(u) = N(\mu \| m_0, V_0)$，则可以推出一个对 $\mu$ 的高斯后验分布，因此可以得到：
$$
\begin{aligned}
p(\mu|D,\Sigma)&= N(\mu|m_N,V_N)\\
V_N^{-1}&= V_0^{-1}+N\Sigma^{-1} \\
m_N&=V_N (\Sigma^{-1}(N\bar x)+V_0^{-1}m_0) \\
\end{aligned}
$$
这就跟基于有噪声的雷达光电来推测目标位置是一模一样的过程，只不过这时候在推测的是一个分布的均值，而不是有噪声的样本。(对于一个贝叶斯方法来说,参数的不确定性和其他任何事情的不确定性没有区别)



可以设置 $V_0 = \infty I$ 来建立一个无信息先验，则有 $p(\mu\|D,\Sigma)=N(\bar x \frac{1}{N}\Sigma)$，所以后验均值就等于最大似然估计，另外，后验方差降低到了 $\frac{1}{N}$，这是频率视角概率统计的标准结果。





## C5. Bayesian statistics

### 5.1 后验分布总结

#### 5.1.1 最大后验估计 (MAP estimation)

下面将讨论最大后验估计存在的缺陷：

1. **无法测量不确定度**：最大后验估计最明显的一个缺陷就在于没有对不确定性提供任何量度，其他的各种点估计比如后验均值或者中位数也都有这个问题。在很多应用中，都需要知道对一个估计值到底能信任多少。

2. **可能导致过拟合**：最大后验分布估计中进行插值可能导致过拟合。在机器学习里面，对于预测准确性往往比模型参数的可解释性更看重。不过如果对参数的不确定度不能建模的话，就可能会导致预测分布存在过拟合问题

3. **众数是非典型点**：

   选择众数作为一个后验分布的总结通常是非常差的选择，因为众数在一个分布通常不典型，不像均值或者中位数那样具有代表意义

   ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051729658.png)

   使用决策理论总结概括一个后验分布将是一个好的选择，其基本思想就是设置一个损失函数，其中的 $L(\theta, \hat \theta)$ 表示的意思是在真实值为 $\theta$ 而估计值为 $\hat \theta$ 时造成的损失。

   * 如果使用 0-1 二值化量化损失，即 $L(\theta, \hat \theta) = I(\theta \neq \hat \theta)$，则最优估计就是后验众数。0-1损失函数就意味着你只使用那些没有误差的点，其他有误差的点将被抛弃（无部分置信）；
   * 对于连续值变量，通常使用平方误差损失函数 $L(\theta, \hat \theta) = (\theta - \hat \theta)^2$，对应的最优估计为后验均值；也可以绝对值损失函数（更健壮）：$L(\theta, \hat \theta) = \|\theta - \hat \theta\|$，对应的最优估计为后验中位数

4. **最大后验估计对重参数化是可变的** *



#### 5.1.2 置信区间

除了点估计之外，我们还经常需要对置信度进行衡量。标准的置信度衡量标准是某个标量值 $\theta$，对应的是后验分布的 "宽度"，这里可以使用一个 $100(1 - \alpha)\%$ 置信区间来衡量，这是一个连续的区域 $C = (l, u)$ ($l, u$ 分别代表下界和上界)，这个区域包含了 $1 - \alpha$ 的后验概率质量，即：
$$
C_{\alpha}(D) = (l, u) : P(l \leq \theta \leq u ) = 1 - \alpha
$$
 可能会有很多个这样的区间，我们需要选一个能满足在每个尾部有 $(1-\alpha)/2$ 概率质量的区间（中央区间）



如果一个后验分布有已知的函数形式，则可以使用 $l = F^{-1}(\alpha / 2), u = F^{-1}(1- \alpha/2)$ 来计算得到后验中央区间，其中 $F$ 是后验的累积分布函数。

> eg. 如果后验是正态分布，$p(\theta\|D) = N(0, 1), \alpha = 0.05$，则可以得到 $l = \Phi(\alpha / 2) = -1.96, u = \Phi(1 - \alpha / 2) = 1.96$，其中 $\Phi$ 为正态分布的累积分布函数，这也证明了实际应用中使用$\mu\pm 2\sigma$做置信区间的可行性，其中的$\mu$表示的是后验均值，$\sigma$表示的是后验标准差，2是对1.96的一个近似。



##### 5.1.2.1 最高后验密度区 (Highest posterior density regions) *



#### 5.1.3 比例差别的推导 (Inference for a difference in proportions)

> 问题背景：假设你要从亚马逊买个东西，然后有两个不同的卖家，提供同样的价格。第一个卖家有90个好评，10个差评,第二个卖家有2个好评没有差评，那你从哪个卖家那里买呢？

使用贝叶斯分析来处理上述问题：

设 $\theta_1, \theta_2$ 分别是两个卖家的可信度，均为未知量，因为没有其他相关信息，这里都使用均匀分布 $\theta_i \sim Beta(1, 1)$，则后验分布为
$$
p(\theta_1|D_1)=Beta(91,11), \ p(\theta_2|D_2)=Beta(3,1)
$$
 我们的计算目标是 $p(\theta_1 > \theta_2 \| D)$，这里定义一个比率的差值 $\delta = \theta_1 - \theta_2$，使用如下的数值积分即可计算目标变量
$$
p(\delta>0|D) =\int^1_0\int^1_0 I(\theta_1>\theta_2)Beta(\theta_1|y_1+1,N_1-y_1+1)Beta(\theta_2|y_2+1,N_2-y_2+1)d\theta_1 d\theta_2
$$
计算结果为 $p(\delta>0\|D)=0.710$，也就意味着最好还是从第一个卖家那里买。

解决这个问题有一个更简单的方法：使用蒙特卡洛方法抽样来估计后验分布 $p(\delta\|D)$，因为在后验中$\theta_1,\theta_2$两者是相互独立的，而且都遵循$\beta$分布，可以使用标准方法进行取样。



### 5.2 贝叶斯模型选择

* 使用交叉验证，估算所有备选模型的泛化误差，然后选择最优模型，不过这需要对每个模型拟合 $K$ 次，$K$ 是交叉验证的折数

* 更有效率的方法是计算不同模型的后验：
  $$
  p(m|D)=\frac{p(D|m)p(m)}{\sum_{m\in M}p(m,D)}
  $$
  然后就很容易计算出最大后验估计模型 $\hat m =\arg \max p(m\|D)$ （贝叶斯模型选择）

  如果对模型使用均匀先验，即 $p(m)\propto 1$，这相当于挑选能够使以下概率最大化的模型:

  $$
  p(D|m)=\int p(D|\theta)p(\theta|m)d\theta
  $$
  

  这个量也叫做模型 $m$ 的边缘似然率、积分似然率或者证据



#### 5.2.1 贝叶斯估计的奥卡姆剃刀

**贝叶斯奥卡姆剃刀效应**

> 如果使用 $p(D\|\hat\theta_m)$ 来选择模型，其中的 $\hat \theta _m$ 是模型 $m$ 参数的最大似然估计或者最大后验估计，则可能总会偏向于选择有最多参数的模型，因为有更多参数的模型会对数据有更好的拟合，因此可以得到更高的似然率，但是如果对参数进行积分，而不是最大化，则可以自动避免过拟合：有更多参数并不必然就有更高的边缘似然率。

* 理解贝叶斯奥卡姆剃刀的一种方式是将边缘似然率改写成以下形式（基于概率论的链式规则）
  $$
  p(D)=p(y_1)p(y_2|y_1)p(y_3|y_{1:2})...p(y_N|y_{1:N-1})
  $$
  

  上式中为了简单起见去掉了关于 $x$ 的条件，这个和留一验证法估计似然率在形式上很相似，因为也是给出了之前的全部点之后预测未来的每个点的位置。



* 另外一种理解贝叶斯奥卡姆剃刀效应的思路是参考概率总和积累起来必然是1。这样就有 $\sum_{D'}p(D'\|m)=1$，其中的求和是在全部可能的数据点集合上进行的。

  * 概率质量守恒原则

    > 复杂的模型可能进行很多预测，就必须把概率质量分散得特别细，然后对任意给定数据就不能得到简单模型一样大的概率。

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051730428.png)

  如上图所示，水平方向的是所有可能的数据集，按照复杂性递增排序 (以某种抽象概念来衡量)。在纵轴上投下的是三个可能的概率模型：$M_1,M_2,M_3$复杂性递增。.实际观测到底数据为竖直线条所示的$D_0$。图示可知,第一个模型太简单了，给$D_0$的概率太低；第三个模型给$D_0$的概率也很低，因为分布的更宽更窄。第二个模型就看上去正好，给已经观测到底数据给出了合理的置信程度，但又没有预测更多。因此第二个模型是最可选的模型。



#### 5.2.2 计算边缘似然率(证据)

当我们讨论推导一个混合模型的参数的时候，通常会写:
$$
p(\theta|D,m)\propto p(\theta|m)p(D|\theta,m)
$$


然后忽略掉归一化常数 $p(D\|m)$，因为 $p(D\|m)$ 相对于 $\theta$ 来说是恒定的，所以这样也有效。不过如果对比模型的话，需要知道如何去计算边缘似然率 $p(D\|m)$。一般来说这个会比较麻烦，因为必须要对所有可能的参数值来进行积分，但是如果有了一个共轭先验，就很容易计算了。

设 $p(\theta)=q(\theta)/Z_0$ 是先验，然后 $q(\theta)$ 是一个未归一化的分布，而$Z_0$是针对这个先验的归一化常数。设 $p(D\|\theta)=q(D\|\theta)/Z_l$ 是似然率，其中的$Z_l$ 包含了似然函数中的任意常数项。最终设 $p(\theta\|D)=q(\theta\|D)/Z_N$ 是后验，其中的 $q(\theta\|D)=q(D\|\theta)q(\theta)$ 是未归一化的后验，而 $Z_N$ 是这个后验的归一化常数，则有:
$$
\begin{aligned}
p(\theta|D)&= \frac{p(D|\theta)p(\theta)}{p(D)} &\text{(5.16)}\\
\frac{q(\theta|D)}{Z_N}&= \frac{q(D|\theta)q(\theta)}{Z_lZ_0p(D)} &\text{(5.17)}\\
p(D)&=\frac{Z_N}{Z_0Z_l}  &\text{(5.18)}\\
\end{aligned}
$$

所以只要归一化常数能算出来,就可以很简单地计算出边缘似然率



##### 5.2.2.1 $\beta$-二项模型(Beta-binomial model)

先把上面的结论用到$\beta$-二项模型上面.已知了$p(\theta\|D)=Beta(\theta\|a',b'), \ a'=a+N_1,b'=b+N_0$。这个后验的归一化常数是 $B(a',b')$。因此有:

$$
\begin{aligned}
p(\theta|D)&= \frac{p(D|\theta)p(\theta)}{p(D)}\\
&= \frac{1}{p(D)}[\frac{1}{B(a,b)}\theta^{a-1}(1-\theta)^{b-1}][\binom{N}{N-1}\theta^{N_1}(1-\theta)^{N_0}] \\
&= \binom{N}{N-1}\frac{1}{p(D)}\frac{1}{B(a,b)}[\theta^{a+N_1-1}(1-\theta)^{b+N_0-1}]\\
\end{aligned}
$$

因此有:

$$
\begin{aligned}
\frac{1}{B(a+N_1,b+N_0)}&= \binom{N}{N-1}\frac{1}{p(D)}\frac{1}{B(a,b)}\\
p(D)&= \binom{N}{N-1}\frac{B(a+N_1,b+N_0)}{B(a,b)}\\
\end{aligned}
$$

$\beta$-伯努利分布模型的边缘似然函数和上面的基本一样，唯一区别就是去掉了$\binom{N}{N-1}$这一项.




##### 5.2.2.2 狄利克雷-多重伯努利模型 (Dirichlet-multinoulli model)

和上面 $\beta$-伯努利模型类似，狄利克雷-多重伯努利模型的边缘似然函数如下所示：
$$
p(D)=\frac{B(N+\alpha)}{B(\alpha)}
$$
其中，$B(\alpha)=\frac{\prod ^K_{k=1}\Gamma(\alpha_k)}{\Gamma(\Sigma_k\alpha_k))}$
把上面两个结合起来写成如下所示形式（重要等式）
$$
p(D)=\frac{\Gamma(\Sigma_k\alpha_k)}{\Gamma(N+\Sigma_k\alpha_k)}\prod_k \frac{\Gamma(N_k+\alpha_k)}{\Gamma(\alpha_k)}
$$


##### 5.2.2.3 高斯-高斯-威沙特分布 (Gaussian-Gaussian-Wishart model)

设想使用了一个共轭正态逆威沙特分布的多元正态分布。设 $Z_0$ 是先验的归一化项，$Z_N$ 是后验的归一化项,$Z_t=(2\pi)^{ND/2}$是似然函数的归一化项，然后很明显就能发现：

$$
\begin{aligned}
p(D)&= \frac{Z_N}{Z_0Z_1}     \\
&=  \frac{1}{\pi^{ND/2}}\frac{1}{2^{ND/2}}\frac{ (\frac{2\pi}{k_N})^{D/2} |S_N|^{-v_N/2}2^{(v_0+N)D/2}\Gamma_D(v_N/2) }{ (\frac{2\pi}{k_0})^{D/2} |S_0|^{-v_0/2}2^{v_0D/2}\Gamma_D(v_0/2)  }    \\
&= \frac{1}{\pi^{ND/2}}( \frac{k_0}{k_N} )^{D/2} \frac{|S_0|^{-v_0/2}\Gamma_D(v_N/2) }{|S_N|^{-v_N/2}\Gamma_D(v_0/2)}  \\
\end{aligned}
$$





##### 5.2.2.4 对数边缘似然函数的贝叶斯信息标准估计 (BIC approximation to log marginal likelihood)

一般来说， 计算 $p(D\|m)=\int p(D\|\theta)p(\theta\|m)d\theta$ 中的积分会比较困难，一种简单又流行的近似方法是使用贝叶斯信息量（Bayesian information criterio，BIC）:
$$
BIC\overset\triangle{=}\log p(D|\hat \theta) -\frac{dof(\hat \theta)}{2}\log N\approx \log p(D) 
$$


上式中的 $dof(\hat\theta)$ 是模型中的自由度个数，而 $\hat\theta$ 是模型的最大似然估计。这有一种类似惩罚对数似然函数的形式，其中的惩罚项依赖于模型的复杂度。



**以一个线性回归为例**

> 最大似然估计为 $\hat w = (X^T X)^{-1}X^Ty$, $\hat\sigma^2= RSS/N$,$RSS=\sum^N_{i=1}(y_i -\hat w^T_{mle}x_i)^2$，对应的对数似然函数为:
>
> $\log p(D\|\hat\theta)=-\frac{N}{2}\log(2\pi\hat\sigma^2)-\frac{N}{2}$
>
> 因此对应的贝叶斯信息量评分为 (去除常数项)：
> $\text{BIC}=-\frac{N}{2}\log (\hat\sigma^2)-\frac{D}{2}\log(N)$
>
> 其中的Ｄ是模型中的变量个数，在统计学中，通常对BIC有另外的一种定义,称之为贝叶斯信息量损失 (BIC cost），目的是将其最小化：
> $\text{BIC-cost}\overset\triangle{=} -2\log p(D\|\hat\theta)+dof(\hat\theta)\log(N)\approx -2\log p(D)$
>
> 在线性回归的情况下,这就变成了:
> $\text{BIC-cost}= N\log(\hat\sigma^2)+D \log (N)$



贝叶斯信息量 (BIC) 方法非常类似于最小描述长度原则 (minimum description length，MDL)，这个原则是根据模型拟合数据的程度以及定义复杂度来对模型进行评分。

还有一个和 BIC/MDL 非常相似的概念叫做赤池信息量 (Akaike information criterion，AIC)，定义如下所示：
$AIC(m,D)\overset\triangle{=}\log p(D\|\hat\theta_{MLE})-dof(m)$

这个概念是从频率论统计学的框架下推导出来的，不能被解释为对边缘似然函数的近似。虽然它的形式和BIC很相似，但是可以看出AIC当中的惩罚项要比BIC里面小，这就导致了AIC会挑选比BIC更复杂的模型，不过这也会导致更好的预测精度。



##### 5.2.2.5 先验的效果

* 在进行后验推导的时候，先验的细节可能不太重要，因为经常是似然率会覆盖了先验。但是在计算边缘似然函数的时候，先验扮演的角色就重要多了，因为要对所有可能的参数设定上的似然函数进行平均，然后用先验来做权重。

* 如果先验未知，则正确的贝叶斯过程是先对先验给出一个先验，也就是对超参数 $\alpha$ 和参数 $w$ 给出一个先验，要计算边缘似然函数，就要对所有未知量进行积分，也就是要计算：
  $$
  p(D|m)=\int\int p(D|w)p(w|\alpha,m)p(\alpha|m)dwd\alpha
  $$

  * 此时需要实现制定一个超先验 (hyper-prior，即对先验的先验)，但是好在在贝叶斯层次中越高的层次就对先验设置越不敏感，所以通常使用无信息的超先验。

  * 计算的一个捷径就是对$\alpha$进行优化，而不是去积分，即使用下面的近似 (经验贝叶斯，empirical Bayes，EB)
    $$
    p(D|m)\approx \int p(D|w)p(w|\hat\alpha,m)dw
    $$
    其中 $\hat\alpha=\arg\max_{\alpha} p(D\|\alpha,m)=\arg\max_{\alpha}\int p(D\|w)p(w\|\alpha,m)dw $



#### 5.2.3 贝叶斯因数（Bayes factors）

* 设模型先验是均匀分布的，即 $p(m)\propto 1$。则此时的模型选择就等价于选择具有最大边缘似然率的模型，现在假设有两个模型可选，分别是空假设 (null hypothesis) $M_0$ 和替换假设 (alternative hypothesis) $M_1$，则贝叶斯因数 = 边缘似然函数之比：

  > $BF_{1,0}\overset\triangle{=}\frac{ p(D\|M_1)}{p(D\|M_0)}=\frac{p(M_1\|D)}{p(M_0\|D)}/\frac{p(M_1)}{p(M_0)}$

  这个跟似然率比值很相似，区别就是将参数整合了进来，因此可以对不同复杂度的模型进行对比。

  * 如果$BF_{1,0}>1$，我们就优先选择模型 1，否则就选择模型 0 

  * 如果$BF_{1,0}$ 仅略大于 1，则此时不能确信模型 1 是更好的选择。Jeffreys 提出了一系列的证据范围来解析贝叶斯因数的不同值，如下表格所示，这个大概就相当于频率论统计学中的 p 值在贝叶斯统计学中的对应概念，或者也可以把这个贝叶斯因数转换成对模型的后验

    ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051731831.png)

  * 如果 $p(M_1)=p(M_0)=0.5$，则有：

    > $p(M_0\|D)=\frac{BF_{0,1}}{1+BF_{0,1}}=\frac{1}{BF_{1,0}+1}$

    

#### 5.2.4 杰弗里斯 - 林德利悖论 (Jeffreys-Lindley paradox) *

​	略





### 5.3 先验 

贝叶斯派认为所有的推测都必须是以客观世界的某些假设为条件的，但是人们还是希望能够尽量缩小事先假设的影响，本节主要介绍实现该目的的几种方法。



#### 5.3.1 无信息先验

如果关于 $\theta$ 应该是啥样没有比较强的事先认识，通常都会使用无信息先验，然后 "let the data speak for itself"

设计一个无信息先验还是需要技巧的。例如，伯努利参数$\theta\in [0,1]$，有人可能会觉得最无信息的先验应该是均匀分布$Beta(1,1)$。但是此时后验均值就是$E[\theta\|D]=\frac{N_1+1}{N_1+N_0+2}$，但是最大似然估计为 $\frac{N_1}{N_1+N_0}$，因此就可以发现这个先验也并不是完全无信息的。

很显然，通过降低伪计数的程度，就可以降低先验的影响。综上所述，最无信息的先验应该是：

>  $\lim_{c\rightarrow 0}Beta(c,c)=Beta(0,0)$

上面这个是在0和1两个位置上有质量的等价点的混合，也叫做Haldane先验，要注意的是，这个Haldane先验是一个不适当先验，也就是积分不为1。



#### 5.3.2 杰弗里斯先验论 (Jeffreys priors)*

​	略

#### 5.3.3 健壮先验 (Robust priors)

在很多情况下,我们对先验并不一定很有信心，所以就需要确保先验不会对结果有过多的影响，这可以通过使用健壮先验来实现，这种先验通常都有重尾，可以避免过分靠近先验均值

**example**

设 $x$ 服从正态分布，即$x\sim N(\theta,1)$，观察到了 $x=5$ 然后要去估计 $\theta$，最大似然估计是 $\hat\theta=5$，看上去也挺合理。在均匀先验之下的后验均值也是这个值，即 $\bar\theta=5$。

不过如果假设我们知道了先验中位数是0，而先验的分位数分别是-1和1，则有$p(\theta \le -1)=p(-1<\theta \le 0)=p(0<\theta \le 1)=p(1<\theta)=0.25$

另外假设这个先验是光滑单峰的，很明显正态分布的先验$N(\theta\|0,2.19^2)$满足这些约束条件。但此时后验均值是3.43，看着就不太让人满意了。

然后再考虑使用柯西分布作先验 $T(\theta\|0,1,1)$，这也满足上面的先验约束条件，这次发现用这个先验的话,后验均值就是4.6,看上去就更合理了。



#### 5.3.4 共轭先验的混合

* 健壮先验很有用，但是计算开销太大

* 共轭先验可以降低计算难度，但对我们把已知信息编码成先验来说，往往又不够健壮，也不够灵活

* 共轭先验的混合还是共轭的，还可以对任意一种先验进行近似，这样的先验能在计算开销和灵活性之间得到一个很不错的折中

  

以抛硬币模型为例，考虑要检查硬币是否作弊，是否两面概率都一样，还是有更大概率人头朝上。

* 此时就不能用一个$\beta$分布来表示了，不过可以把它用两个$\beta$分布的混合来表示。例如，可以使用:

  > $p(\theta)=0.5Beta(\theta\|20,20)+0.5Beta(\theta\|30,10)$

  如果 $\theta$ 来自第一个分布,就说明没作弊，如果来自第二个分布，就说明有更大概率人头朝上



* 可以引入一个潜在指示器变量 $z$ 来表示这个混合，其中的 $z=k$ 的意思就是 $\theta$ 来自混合成分 $k$，则先验具有以下形式：

  > $p(\theta)=\sum_k p(z=k)p(\theta\|z=k)$

  其中的每个 $p(\theta\|z=k)$ 都是共轭的，而 $p(z=k)$ 就叫做先验的混合权重 (prior mixing weights)，

  

* 此外，后验也可以写成一系列共轭分布的混合形式：

  > $p(\theta\|D)=\sum_k p(z=k)p(\theta\|D,z=k)$

  其中的 $p(Z=k\|D)$ 是后验混合权重 (posterior mixing weights)，如下所示:

  > $p(Z=k\|D)=\frac{p(Z=k)p(D\|Z=k)}{\sum_{k'}p(Z=k')p(D\|Z=k')}$

  这里的 $p(D\|Z=k)$ 是混合成分k的边缘似然函数





##### Example

假如使用下面的混合先验：
$p(\theta)=0.5 \ \text{Beta}(\theta\|a_1,b_1)+0.5 \ \text{Beta}(\theta\|a_2,b_2)$

其中$a_1=b_1=20,a_2=b_2=10$。然后观察了$N_1$次的人头，$N_0$次的背面，后验就变成:

$p(\theta\|D)=p(Z=1\|D) \ \text{Beta}(\theta\|a_1+N_1,b_1+N_0)+p(Z=2\|D) \ \text{Beta}(\theta\|a_2+N_1,b_2+N_0)$

如果$N_1=20,N_0=10$，那么使用以下等式：

> $p(D)= \binom{N}{N-1}\frac{B(a+N_1,b+N_0)}{B(a,b)}$

即可得到了以下后验：

> $p(\theta\|D)=0.346 \ \text{Beta}(\theta\|40,30)+0.654 \ \text{Beta}(\theta\|50,20)$

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051731797.png)



### 5.4 分层贝叶斯(Hierarchical Bayes)

计算后验$p(\theta\|D)$的一个关键要求就是特定的先验 $p(\theta\|\eta)$，其中的$\eta$是超参数，设置 $\eta$ 的两种方法：

* 有的时候可以使用无信息先验

* 一个更加贝叶斯风格的方法就是对先验设一个先验，可以用下面的方式来表达:

  > $\eta \rightarrow \theta \rightarrow D$

  这就是一个分层贝叶斯模型，也叫作多层模型，因为有多层的未知量.

​	

#### 5.4.1 样例：与癌症患病率相关的模型

考虑在不同城市预测癌症患病率的问题，具体来说，加入我们要测量不同城市的人口 $N_i$，然后对应城市死于癌症的人口数$x_i$。假设$x_i\sim Bin(N_i,\theta_i)$，然后要估计癌症发病率$\theta_i$。

* 一种方法是分别进行估计，不过这就要面对稀疏数据问题（低估了人口少即$N_i$小的城市的癌症发病率）；

* 另外一种方法是假设所有的 $\theta_i$ 都一样，这叫做参数绑定，结果得到的最大似然估计正好就是$\hat\theta =\frac{\Sigma_ix_i}{\Sigma_iN_i}$。可是很明显假设所有城市癌症发病率都一样有点过于牵强了。

* 有一种折中的办法，就是估计$\theta_i$是相似的，但可能随着每个城市的不同而又发生变化。这可以通过假设$\theta_i$服从某个常见分布来实现，比如$\beta$分布，即$\theta_i\sim Beta(a,b)$。这样就可以把完整的联合分布写成下面的形式：
  $$
  p(D,\theta,\eta|N)=p(\eta)\prod^N_{i=1}Bin(x_i|N_i,\theta_i)Beta(\theta_i|\eta)
  $$
  

上式中的$\eta=(a,b)$，要注意这里很重要的一点是要从数据中推测 $\eta=(a,b)$；如果只是随便设置成一个常数，那么$\theta_i$就会是有条件独立的，在彼此之间就没有什么信息联系了。与之相反的，若将$\eta$完全看做一个未知量（隐藏变量），就可以让数据规模小的城市从数据规模大的城市借用统计强度。

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051732867.png)

* 要计算联合后验$p(\eta,\theta\|D)$，从这里面可以得到后验边缘分布$p(\theta_i\|D)$，如上图 (a) 所示，图中的蓝色柱状是后验均值 $E[\theta_i\|D]$，红色线条是城市人口均值 $E[a/(a+b_\|D]$ （这代表了$\theta_i$的均值)。很明显可以看到后验均值朝向有小样本$N_i$的城市的汇总估计方向收缩。例如，城市1和城市20都观察到有0的癌症发病率，但城市20的人口数较少，所以其癌症发病率比城市1更朝向人口估计方向收缩（也就是距离水平的红色线更近）.

* 上图 (b) 中展示的是 $\theta_i$ 的95%后验置信区间，可以看到城市15有特别多的人口(53637)，后验不确定性很低。所以这个城市对 $\eta$ 的后验估计的影响最大，也会影响其他城市的癌症发病率的估计。城市10和19有最高的最大似然估计，也有最高的后验不确定性，反映了这样高的估计可能和先验相违背（先验视从所有其他城市估计得到的）.

上面这个例子中，每个城市都有一个参数，然后对相应的概率进行建模，通过设置伯努利分布的频率参数为一个协变量的函数，即$\theta_i=sigm(w^T_ix)$，就可以对多个相关的逻辑回归任务进行建模了。这也叫做多任务学习 (multi-task learning)。



### 5.5 经验贝叶斯(Empirical Bayes)

在分层贝叶斯模型中，我们需要计算多层的潜在变量的后验。例如，在一个两层模型中，需要计算：

> $p(\eta,\theta\|D)\propto p(D\|\theta)p(\theta\|\eta)p(\eta)$

有的时候可以通过分析将 $\theta$ 边缘化，将问题简化成只去计算$p(\eta\|D)$.

作为计算上的简化，可以对超参数后验进行点估计来近似，即 $p(\eta\|D)\approx \delta_{\hat\eta}(\eta)$，其中的$\hat\eta=\arg\max p(\eta\|D)$。因为 $\eta$ 通常在维数上都比 $\theta$ 小很多，这个模型不太容易过拟合，所以我们可以安全地来对 $\eta$ 使用均匀先验，因此估计就变成了:

$$
\hat\eta =\arg\max p(D|\eta)=\arg\max \left[\int p(D|\theta)p(\theta|\eta)d\theta \right]
$$
其中括号里面的量就是边缘似然函数或整合似然函数，也叫证据。这个方法总体上叫做经验贝叶斯，也叫做第二类最大似然估计，在机器学习里面，也叫作证据程序。

**NOTE**：经验贝叶斯违反了先验应该独立于数据来选择的原则，不过可以将其视作是对分层贝叶斯模型中的推导的一种近似，计算开销更低，类比于最大后验估计可以看作是对单层模型 $\theta\rightarrow D$ 的推导的近似一样。实际上,可以建立一个分层结构，其中进行的积分越多，就越"贝叶斯化":



| Method                                           | Definition                                                   |
| ------------------------------------------------ | ------------------------------------------------------------ |
| 最大似然估计 (Maximum Likelihood)                | $\hat\theta=\arg\max\limits_\theta p(D\|\theta)$             |
| 最大后验估计 (MAP estimation)                    | $\hat\theta=\arg\max_\theta p(D\|\theta)p(\theta\|\eta)$     |
| 经验贝叶斯的最大似然估计 (ML-II Empirical Bayes) | $\hat\theta=\arg\max_\eta \int p(D\|\theta)p(\theta\|\eta)d\theta=\arg\max p(D\|\eta)$ |
| 经验贝叶斯的最大后验估计 (MAP-II)                | $\hat\theta=\arg\max_\eta \int p(D\|\theta)p(\theta\|\eta)p(\eta)d\theta=\arg\max p(D\|\eta)$ |
| 全贝叶斯 (Full Bayes)                            | $p(\theta,\eta\|D)\propto p(D\|\theta)p(\theta\|\eta)p(\eta)$ |





#### 5.5.1 样例：$\beta$-二项模型

回到癌症发病率的模型上，可以积分掉$\theta_i$，然后直接写出边缘似然函数，如下所示：

$$
\begin{aligned}
p(D|a,b)&=\prod_i \int Bin(x_i|N_i,\theta_i)Beta(\theta_i|a,b)d\theta_i  \\
&=\prod_i \frac{B(a+x_i,b+N_i-x_i)}{B(a,b)}  \\
\end{aligned}
$$

*a 和 b 最大化的方法参考 (Minka 2000e)*

估计完了 a 和 b 之后，就可以代入到超参数里面来计算后验分布 $p(\theta_i\|\hat a,\hat b,D)$，还按照之前的方法，使用共轭分析，得到的每个$\theta_i$的后验均值就是局部最大似然估计和先验均值的加权平均值，依赖于$\eta=(a,b)$；但由于$\eta$是根据所有数据来估计出来的，所以每个$\theta_i$也都受到全部数据的影响。



#### 5.5.2 样例:高斯-高斯模型 (Gaussian-Gaussian model)

下面的例子和癌症发病率的例子相似，不同之处是这个例子中的数据是实数值的，使用一个高斯(正态)似然函数和一个高斯(正态)先验，这样就能写出来解析形式的解。

假设拥有来自多个相关群体的数据，比如$x_{ij}$表示的就是学生i在学校j得到的测试分数，$j$的取值范围从1到D，而$i$是从1到$N_j$，即$j=1:D,i=1:N_j$，然后想要估计每个学校的平均分$\theta_j$，可是样本规模$N_j$对于一些学校来说可能很小，所以可以用分层贝叶斯模型来规范化这个问题，即假设$\theta_j$来自一个常规的先验 $N(\mu,\tau^2)$.

这个联合分布的形式如下所示：
$$
p(\theta,D|\eta,\sigma^2)=\prod^D_{j=1}N(\theta_j|\mu,\tau^2)\prod^{N_j}_{i=1}N(x_{ij}|\theta_j,\sigma^2)
$$
为了简化问题，上式中假设了$\sigma^2$是已知的。

接下来将估计$\eta$，一旦估计了 $\eta=(\mu,\tau)$，就可以计算$\theta_j$的后验。要进行这个计算，只需要将联合分布改写成下面的形式，这个过程利用了值$x_{ij}$和方差为$\sigma^2$的$N_j$次高斯观测等价于值为$\bar x_j \overset\triangle{=} \frac{1}{N_j}\sum^{N_j}_{i=1}x_{ij}$方差为$\sigma^2_j\overset\triangle{=}\sigma^2/N_j$的一次观测这个定理，则可得到了:
$$
p(\theta,D|\hat\eta,\sigma^2)=\prod^D_{j=1}N(\theta_j|\hat\mu,\hat\tau^2)N(\bar x_j|\theta_j,\sigma^2_j)
$$


结合上面的式子以及**定理4.4.1**，即可得到后验为:

$$
\begin{aligned}
p(\theta_j|D,\hat\mu,\hat\tau^2)&= N(\theta_j|\hat B_j\hat\mu+(1-\hat B_j)\bar x_j,(1-\hat B_j)\sigma^2_j)  \text{(5.84)}\\
\hat B_j &\overset\triangle{=}  \frac{\sigma^2_j}{\sigma^2_j+\hat\tau^2}\text{(5.85)}\\
\end{aligned}
$$

其中的$\hat\mu =\bar x,\hat\tau^2$下面会给出定义.

* $0\le \hat B_j \le 1$这个量控制了朝向全局均值 $\mu$ 的收缩程度，如果对于第$j$组来说数据可靠（比如可能是样本规模$N_j$特别大），那么 $\sigma^2_j$ 就会比 $\tau^2$ 小很多，因此 $\hat B_j$ 也会很小，然后就会在估计 $\theta_j$ 的时候给 $\bar x_j$ 更多权重。而样本规模小的群组就会被规范化，也就是朝向全局均值$\mu$的方向收缩更严重

* 如果对于所有的组 $j$ 来说都有$\sigma_j=\sigma$，那么后验均值就成了：

  > $\hat\theta_j= \hat B\bar x+(1-\hat B)\bar x_j=\bar x +(1-\hat B)(\bar x_j-\bar x) $

  这和后将会讨论的吉姆斯-斯坦因估计器 (James Stein estimator) 的形式一样

​		



##### 5.6.2.1 样例：预测棒球得分

接下来这个例子是把上面的收缩(shrinkage)方法用到棒球击球平均数(baseball batting averages, 引自 Efron and Morris 1975).观察D=18个球员在前T=45场比赛中的的击球次数.把这个击球次数设为$b_i$.假设服从二项分布,即$b_j\sim Bin(T,\theta_j)$,其中的$\theta_j$是选手j的"真实"击球平均值.目标是要顾及出来这个$\theta_j$.最大似然估计(MLE)自然是$\hat\theta_j=x_j$,其中的$x_j=b_j/T$是经验击球平均值.不过可以用经验贝叶斯方法来进行更好的估计.

要使用上文中讲的高斯收缩方法(Gaussian shrinkage approach),需要似然函数是高斯分布的,即对于一直的$\sigma^2$有$x_j\sim N(\theta_j,\sigma^2)$.(这里去掉了下标i因为假设了$N_j=1$而$x_j$已经代表了选手j的平均值了.)不过这个例子里面用的是二项似然函数.均值正好是$E[x_j]=\theta_j$,方差则是不固定的:

$var[x_j]=\frac{1}{T^2}var[b_j]=\frac{T\theta_j(1-\theta_j)}{T^2}$(5.87)

所以咱们对$x_j$应用一个方差稳定变换(variance stabilizing transform 5)来更好地符合高斯假设:
$y_i=f(y_i)=\sqrt{T}\arcsin (2y_i-1)$(5.88)

然后应用一个近似$y_i\sim N(f(\theta_j),1)=N(\mu_j,1)$.以$\sigma^2=1$代入等式5.86来使用高斯收缩对$\mu_j$进行估计,然后变换回去,就得到了:

$\hat\theta_j=0.5(\sin(\hat\mu_j/\sqrt{T})+1)$(5.89)



此处参考原书图5.12

这个结果如图5.12(a-b)所示.在图(a)中,投图的是最大似然估计(MLE)$\hat\theta_j$和后验均值$\bar\theta_j$.可以看到所有的估计都朝向全局均值0.265收缩.在图(b)中,投图的是$\theta_j$的真实值,最大似然估计(MLE)$\hat\theta_j$和后验均值$\bar\theta_j$.(这里的$\theta_j$的真实值是指从更大规模的独立赛事之间得到的估计值.)可以看到平均来看,收缩的估计比最大似然估计(MLE)更加靠近真实值.尤其是均方误差,定义为$MSE=\frac{1}{N}\sum^D_{j=1}(\theta_j-\bar\theta_j)^2$,使用收缩估计的$\bar\theta_j$比最大似然估计的$\hat\theta_j$的均方误差小了三倍.




##### 5.6.2.2 估计超参数

在本节会对估计$\eta$给出一个算法.加入最开始对于所有组来说都有$\sigma^2_j=\sigma^2$.这种情况下就可以以闭合形式(closed form)来推导经验贝叶斯估计(EB estimate).从等式4.126可以得到:
$p(\bar x_j\|\mu,\tau^2,\sigma^2)=\int N(\bar x_j\|\theta_j,\sigma^2)N(\theta_j\|\mu,\tau^2)d\theta_j =N(\bar x_j\|\mu,\tau^2+\sigma^2)$(5.90)

然后边缘似然函数(marginal likelihood)为:

$p(D\|\mu,\tau^2,\sigma^2)=\prod^D_{j=1}N(\bar x_j\|\mu,\tau^2+\sigma^2)$(5.91)

接下来就可以使用对正态分布(高斯分布)的最大似然估计(MLE)来估计超参数了.例如对$\mu$就有:

$\hat \mu =\frac{1}{D}\sum^D_{j=1}\bar x_j=\bar x$(5.92)

上面这个也就是全局均值.

对于方差,可以使用矩量匹配(moment matching,相当于高斯分布的最大似然估计):简单地把模型方差(model varianc)等同于经验方差(empirical variance):

$\hat \tau^2+\sigma^2 =\frac{1}{D}\sum^D_{j]1}(\bar x_j-\bar x)^2\overset\triangle{=} s^2$(5.93)

所以有$\hat \tau^2=s^2-\sigma^2$.因为已知了$\tau^2$必然是正的,所以通常都使用下面这个修订过的估计:

$\hat \tau^2=\max(0,s^2-\sigma^2)=(s^2-\sigma^2)_{+}$(5.94)

这样就得到了收缩因子(shrinkage factor):

$\hat B=\frac{\sigma^2}{\sigma^2+\tau^2}=\frac{\sigma^2}{\sigma^2+(s^2-\sigma^2)_{+}}$(5.95)

如果$\sigma^2_j$各自不同,就没办法以闭合形式来推导出解了.练习11.13讨论的是如何使用期望最大化算法(EM algorithm)来推导一个经验贝叶斯估计(EB estimate),练习24.4讨论了如何在这个分层模型中使用全贝叶斯方法.




### 5.6 贝叶斯决策规则 (Bayesian decision rule)

本节的目标就是设计一个决策程序或者决策策略 $\delta:X\rightarrow A$，对每个可能的输入指定最优行为，使损失函数期望最小：

$$
\delta(x)=\arg\min_{a\in A} E[L(y,a)]
$$

* 在经济学领域,更常见的属于是效用函数，就是将损失函数取负值，即$U(y,a)=-L(y,a)$，这样上面的规则就变成以下形式：

  > $\delta(x)=\arg\max_{a\in A} E[U(y,a)]$ （期望效用最大化规则）

* 在贝叶斯决策理论的方法中，观察了 $x$ 之后的最优行为定义是能够让后验期望损失最小的行为

  > $\rho(a\|x)\overset\triangle{=} E_{p(y\|x)} [L(y,a)]=\sum_y L(y,a)p(y\|x)$

  * 如果 $y$ 是连续的，比如想要估计一个参数向量的时候，应该把上面的求和替换成为积分

  * 这样就有了贝叶斯估计器，也叫做贝叶斯决策规则：

    > $\delta(x)=\arg\min_{a\in A}\rho(a\|x)$

  

  

#### 5.6.1 常见损失函数的贝叶斯估计器

本节将展示对一些机器学习中常遇到的损失函数如何构建贝叶斯估计器

##### 5.6.1.1 最大后验估计最小化0-1损失

0-1损失 (0-1 loss) 的定义如下：
$$
L(y,a)=I(y\ne a)=\begin{cases} 0 &\text{if} &a=y\\ 1 &\text{if} &a\ne y\end{cases}
$$


这通常用于分类问题中，其中的 $y$ 是真实类标签，而 $a=\hat y$ 是估计得到的类标签

例如，在二分类情况下，可以写成下面的损失矩阵：

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051732334.png)



* 后验期望损失为：

  > $\rho(a\|x)=p(a\ne y\|x)=1-p(y\|x)$

* 因此能够最小化期望损失的行为就是后验众数或者最大后验估计：

  > $y^\star(x)=\arg\max \limits_{y\in Y} p(y\|x)$



![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051732232.png)



##### 5.6.1.2 拒绝选项 (Reject option)

在分类问题中，$p(y\|x)$是非常不确定的，所以我们可能更倾向去选择一个拒绝行为 (reject action)，即拒绝将这个样本分类到任何已有的指定分类中，而是告知 "不知道"。

使用正规化语言对拒绝选项进行表达：

假设选择一个$a=C+1$对应的就是选择了拒绝行为，然后选择$a\in \{1,...,C\}$对应的就是分类到类标签中去，然后就可以定义下面的损失函数：
$$
L(y=j,a=i)=\begin{cases} 0 &\text{if} &i=j & i,j \in\{1,...,C\}\\ \lambda_r &\text{if} &i=C+1 \\ \lambda_s &\text{otherwise}\end{cases}
$$





##### 5.6.1.3 后验均值最小化 $l_2$ (二次)损失函数

对于连续参数，更适合使用的损失函数是平方误差函数（ 也叫$l_2$ 损失函数 or 二次损失函数），定义如下:

$$
L(y,a)=(y-a)^2
$$


* 后验期望损失为：
  $$
  \rho(a|x)=E[(y-a)^2|x|]=E[y^2|x]-2aE[y|x]+a^2
  $$
  
* 最优估计即为后验均值：
  $$
  \frac{\partial}{\partial a}\rho(a|x)= -2E[y|x]+2a=0  \Longrightarrow \hat y=E[y|x]=\int y p(y|x)dy
  $$
  

*这也叫做最小均值方差估计（minimum mean squared error，MMSE)*



在线性回归问题中有：

> $p(y\|x,\theta)=N(y\|x^Tw,\sigma^2)$

此时给定某个训练集D之后的最优估计就是：

>$E[y\|x,D]=x^TE[w\|D]$

即后验均值参数估计代入

**NOTE**：注意不论对 $w$ 使用什么样的先验，这都是最优选择




##### 5.6.1.4 后验中位数最小化 $l_1$ (绝对）损失函数

$l_2$（二次）损失函数以二次形式惩罚与真实值的偏离，因此对异常值特别敏感，因此有一个更健壮的替换选择：绝对损失函数，也叫做 $l_1$ 损失函数：

> $L(y,a)=\|y-a\|$

此时的最优估计就是后验中位数，即使得$P(y < a\|x) = P(y \ge a\|x) = 0.5$的 $a$ 值

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051733540.png)



##### 5.6.1.5 监督学习(Supervised learning)

假设有一个预测函数$\delta: X\rightarrow Y$，然后假设有某个损失函数 $l(y,y')$，这个损失函数给出了预测值是 $y'$ 而真实值是 $y$ 的时候的损失，则可以定义采取行为$\delta$（比如使用这个预测器）而未知自然状态为$\theta$（数据生成机制的参数）的时候的损失：（泛化误差）

$$
L(\theta,\delta)\overset\triangle{=} E_{(x,y) \sim p(x,y|\theta)}[l(y,\delta(x))]=\sum_x\sum_y L(y,\delta(x))p(x,y|\theta)
$$


我们的目标是最小化后验期望损失，即：
$$
\rho(\delta|D)=\int p(\theta|D)L(\theta,\delta)d\theta
$$


#### 5.6.2 假阳性和假阴性的权衡

针对二分类决策问题，会存在两种错误预测类型：

* 假阳性（false positive, FP）：估计值 $\hat y = 1$，真实值 $y = 0$
* 假阴性（false negative, FN）：估计值 $\hat y = 0$，真实值 $y = 1$



0-1损失函数会同等对待这两种错误，可以用下面这个更通用的损失矩阵来表征这种情况：

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051733254.png)

其中，$L_{FN}$就是假阴性的损失，$L_{FP}$是假阳性的损失，两种可能行为的后验期望损失为:
$$
\begin{aligned}
\rho(\hat y=0|x)&= L_{FN} \ p(y=1|x) \\
\rho(\hat y=1|x)&= L_{FP} \ p(y=0|x) \\
\end{aligned}
$$

因此应选 $\hat y=1$ 当且仅当：

$$
\begin{aligned}
\rho(\hat y=0|x) &>  \rho(\hat y=1|x)\\
\frac{p(y=1|x) }{p(y=0|x) }&>\frac{L_{FP}}{L_{FN}}  \\
\end{aligned}
$$


如果$L_{FN}=cL_{FP}$，选择 $\hat y=1$ 的决策阈值为 $p(y=1\|x) > \tau $，其中的 $\tau=1/(1+c)$ (or $p(y=0 \|x) < c/(c+1)$)。



##### 5.7.2.1 ROC 曲线以及相关内容

ROC曲线提供了学习 FP-FN 权衡的一种方式，而不用必须去选择特定的阈值设置.

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051734126.png)

> $N_+$是真阳性个数，$\hat N_{+}$是预测阳性个数，$N_-$是真阴性个数，$\hat N_{-}$是预测阴性个数



**混淆矩阵**：

|            | $y=1$                                           | $y=0$                          |
| ---------- | ----------------------------------------------- | ------------------------------ |
| $\hat y=1$ | $TP/N_+=TPR= \text{sensitivity}= \text{recall}$ | $FP/N_− =FPR=\text{type I}$    |
| $\hat y=0$ | $FN/N_+ =FNR=\text{miss rate}= \text{type II}$  | $TN/N_− =TNR= \text{speciﬁty}$ |

**ROC曲线**：受试者工作特征曲线 (receiver operating characteristic curve)

通过使用一系列的阈值来运行预测器，然后绘制出 $TPR$ 关于 $FPR$ 的曲线，作为 $\tau$ 的隐含函数

* 一个ROC曲线的质量通常用一个单一数值来表示，也就是曲线所覆盖的面积（area under the curve，缩写为AUC），AUC分数越高就越好
  * AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器。
  * 0.5 < AUC < 1：优于随机预测
  * AUC = 0.5：随机预测，没有预测价值
  * AUC < 0.5：比随机预测还差，但只要反预测而行，就优于随机预测 
* 另外一个统计量是相等错误率（equal error rate，EER），也叫做交错率（cross over rate），定义为满足 FPR = FNR 的值，由于FNR=1-TPR，所以可以画一条线从左上角到右下，然后查看与ROC曲线的交点，EER分数越低越好，最小为0。

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051734045.png)





##### 5.7.2.2 准确率 - 召回率曲线 (Precision recall curves)

> 探究小概率的罕见事件的时候，阴性结果的数量会非常大。此时再去对比真阳率 $TPR = TP/N_+$ 和假阳率 $FPR = FP/N_−$ 就没有参考意义了（因为假阳率 FPR 肯定会很小）。因此在ROC曲线上的所有行为都会出现在最左边，这种情况下，通常就把真阳率 TPR 和假阳性个数绘成一个曲线，而不是使用假阳性率FPR。

* 准确率，precision = $TP / \hat N_+ = TP / (TP + FP) = p(y=1\|\hat y=1)$
  * 准确率衡量的是预测的阳性中有多大比例是真正的阳性
* 召回率，recall = $TP/N_+ = TP/(TP + FN) = p(\hat y=1\|y=1)$
  * 召回率衡量的是阳性中有多少被正确预测到

如果 $\hat y_i \in \{0,1\}$ 是预测的分类标签，而 $y_i \in \{0,1\}$ 是真实分类标签，就可以估计准确率和召回率：
$$
P=\frac{\sum_i y_i\hat y_i}{\sum_i \hat y _i},  R= \frac{\sum_iy_i\hat y_i}{\sum_i y_i}
$$


准确率-召回率曲线就是随着阈值参数 $\tau$ 的变化对准确率与召回率之间的关系绘制得到的曲线（曲线越往右上角，预测效果越好）

* 准确率-召回率曲线可以用一个单一数值来概括，即平均准确率（mean precision，综合所有召回率条件下的准确率），近似等于曲线下的面积；或者也可以用固定召回率下的准确率来衡量，比如在前 K=10 个识别到的项目中的准确率，这称为在K分数的平均准确率.



## C6. Frequentist statistics

### 6.1 概论

* **频率论统计学**

  不是基于后验分布，而是基于抽样分布，这种分布中，估计器在用于不同的数据集的时候，从真实的未知分布中进行抽样，重复试验的变化的概念就构成了使用频率论方法来对不确定性建模的基础。

* **贝叶斯统计学**

  相比之下，贝叶斯方法中只接受被实际观察的数据，而并没有重复测试的概念，这就允许贝叶斯方法用于计算单次事件的概率；另一方面可能更重要的是贝叶斯方法能避免一些困扰了频率论方法的悖论。



### 6.2 一个估计器的抽样分布

在频率论统计学中，参数估计 $\hat\theta$ 是通过对某个数据集 D 来使用一个估计器 $\delta$ 来计算得到的，也就是$\hat\theta=\delta(D)$。这里参数被当做固定的，而数据可以是随机的（正好和贝叶斯方法中的完全相反）。参数估计的不确定性可以通过计算估计器的抽样分布来衡量。



#### 6.2.1 Bootstrap

Bootstrap是一种简单的蒙特卡罗方法，用来对抽样分布进行近似，在估计器是真实参数的复杂函数的情况下特别有用。

**idea**

* 如果我们知道了真实参数$\theta^*$，就可以生成S个假的数据集，每个数据集规模都是N，都是来自于真实分布 $x^s_i \sim p(*\|\theta^*)$，其中$s=1:S,i=1:N$ （重复抽样）
* 从每个样本中计算估计值 $\hat\theta^s =f(x^s_{1:N})$，然后使用所得样本的经验分布作为我们对抽样分布的估计
  * **参数化Bootstrap：**由于$\theta$是未知的，则使用 $\hat\theta(D)$ 作为替代来生成样本
  * **非参数化Bootstrap：**对$x_i^s$从原始数据 D 中进行有放回抽样，然后按照之前的方法来计算诱导分布



**Bootstrap** $\theta^s =\hat\theta(x^s_{1:N})$ VS **后验分布抽样** $\theta^s\sim p(*\|D)$

* 概念上两者很不一样，不过一般情况下，在先验不是很强的时候,这两者可能很相似
* Bootstrap可能会比后验取样要慢很多，原因在于 Bootstrap 必须对模型拟合 S 次，而在后验抽样中，通常只要对模型拟合一次（来找到局部众数），然后就可以在众数周围进行局部探索，这种局部探索通常要比从头拟合模型快得多



#### 6.2.2 最大似然估计的大样本理论 *

定义一个得分函数，即对数自然函数在某一点$\theta$处的梯度：
$$
s(\hat\theta)\overset{\triangle}{=} \nabla \log p(D|\theta)|_{\hat\theta} 
$$
把负得分函数定义成观测信息矩阵，等价于负对数似然函数的海森矩阵：
$$
J(\hat\theta(D))\overset{\triangle}{=} -\nabla s(\hat\theta)=-\nabla^2_\theta \log p(D|\theta)|_{\hat \theta}
$$
在一维情况下：
$$
J(\hat\theta(D))=-\frac{d}{d\theta^2}\log p(D|\theta)|_{\hat\theta}
$$
这是对对数似然函数在点 $\hat \theta$ 位置曲率的一种度量



由于要研究的是抽样分布，$D=(x_1,...,x_N)$是一系列随机变量的集合，则费舍信息矩阵的定义就是观测信息矩阵的期望值：
$$
I_N(\hat\theta|\theta^*) \overset{\triangle}{=}  \mathrm{E}_{\theta^*}[J(\hat\theta|D)]
$$
其中

* $ \mathrm{E}_{\theta^*}[f(D)] \overset{\triangle}{=} \frac{1}{N} \sum^N_{i=1}f(x_i)p(x_i\|\theta^*)$是函数 $f$ 用于从 $\theta^*$ 中取样的数据时的期望值
* 通常这个 $\theta^*$ 表示的都是生成数据的 ”真实参数“，假设为已知的，所以就可以简写为 $I_N(\hat\theta)\overset{\triangle}{=} I_N(\hat\theta\|\theta^*)$
* 容易就能看出$I_N(\hat\theta)=NI_1(\hat\theta)$，因为规模为 N 的样本对数似然函数自然要比规模为 1 的样本更加 "陡峭"，所以可以只写成 $I(\hat\theta) \overset \triangle {=}I_1(\hat\theta)$



设最大似然估计为 $\hat\theta \overset{\triangle}{=}\hat\theta_{mle}(D)$，其中的$D\sim\theta^*$，随着$N \rightarrow \infty$，则有：

$$
\hat\theta \rightarrow N((\theta^*,I_N(\theta^*)^{-1})
$$
则称这个最大似然估计的抽样分布是渐进正态的.

方差可以用来衡量对最大似然估计的信心量度，但是由于$\theta^*$是未知的，所以不能对抽样分布的方差进行估计。我们可以用$\hat\theta$替代$\theta^*$来估计抽样分布，这样得到的$\hat\theta_k$近似标准差为：

$$
\hat{se}_k \overset{\triangle}{=} I_N(\hat\theta)_{kk}^{-\frac{1}{2}} 
$$


### 6.3 频率论决策理论 (Frequentist decision theory)

在频率论方法中，可以自由选择任意的估计器或者决策规则 $\delta:X\rightarrow A$。选好了估计器，就可以定义对应的期望损失或者风险函数：
$$
R(\theta^*,\delta)\overset{\triangle}{=} \mathrm{E} _{p(\tilde D|\theta^*)}[L(\theta^*,\delta(\tilde D))=\int L(\theta^*,\delta(\tilde D))p(\tilde D|\theta^*)d\tilde D]
$$
上式中的$\tilde D$是从 "自然分布" 抽样的数据，用参数$\theta^*$来表示，也就是说，期望值是与估计器的取样分布相关的

与贝叶斯后验期望损失相比：
$$
\rho(a|D,\pi) \overset{\triangle}{=}  \mathrm{E}[L(\theta,a)]=\int_\Theta L(\theta,a)p(\theta|D,\pi)d\theta
$$


很明显贝叶斯方法是在位置的$\theta$上进行平均，条件为已知的D；而频率论方法是在$\tilde D$上平均(也就忽略了观测值)，而条件是未知的$\theta^*$。



#### 6.3.1 贝叶斯风险

目标是把$R(\theta^*,\delta)$转换成一个不需要依赖$\theta^*$的单独量$R(\delta)$，一种方法是对$\theta^*$设一个先验，然后定义一个估计器的贝叶斯风险或者积分风险：
$$
R_B(\delta) \overset{\triangle}{=}  \mathrm{E}_{p(\theta^*)}[R(\theta^*,\delta)]=\int R(\theta^*,\delta)p(\theta^*)d \theta^* 
$$
贝叶斯估计器或者贝叶斯决策规则就是将期望风险最小化：
$$
\delta_B \overset{\triangle}{=} \arg\min_\delta R_B(\delta) 
$$
这里的积分风险函数也叫做预制后验风险（因为是在看到数据之前得到的）



##### **定理 6.3.1** （结合贝叶斯方法和频率论方法）

> **贝叶斯估计器可以通过最小化每个 $x$ 的后验期望损失来得到**

**证明：**

通过切换积分顺序可以得到：
$$
\begin{aligned}
R_B(\delta)& = \int \left[\sum_x\sum_y L(y,\delta(x))p(x,y|\theta^*)\right]p(\theta^*)d\theta^* \\
&=\sum_x\sum_y \int_\Theta L(y,\delta(x))p(x,y,\theta^*)d\theta^* \\
& =\sum_x \left[\sum_y L(y,\delta(x))p(y|x)dy \right]p(x)\\
& =\sum_x \rho (\delta(x)|x)p(x)\\
\end{aligned}
$$
要最小化全局期望，只要将每个 $x$ 项最小化就可以了，所以决策规则为：
$$
\delta_B (x)=\arg\min_{a\in A} \rho(a|x) 
$$




##### **定理 6.3.2**

> 每个可接受的决策规则都是某种程度上的贝叶斯决策规则，对应着某些可能还不适当的先验分布。

这表明对频率论风险函数最小化的最佳方法就是贝叶斯方法



#### 6.3.2 最小最大风险 (Minimax risk)

除了贝叶斯风险，另外一种方法如下所示：

* 定义一个估计器的最大风险：
  $$
  R_{max}(\delta) \overset \triangle{=} \underset{\theta^{\star}} {max} R(\theta ^{\star}, \delta)
  $$

* 最小最大规则就是将最大风险最小化：
  $$
  \delta_{MM} \overset \triangle {=} \underset{\delta}{\text{argmin}}\ R_{max}(\delta) 
  $$

#### 6.3.3 可容许的估计器 (Admissible estimators)

频率论决策理论的基本问题就是要知道真实分布 $p(*\|\theta^*)$ 才能去评估风险。但是有的估计器可能不管$\theta^*$是什么值，都会比其他的一些估计器更差

* 如果对于所有的$\theta\in\Theta$，都有$R(\theta,\delta_1) \leq R(\theta,\delta_2)$，则称 $\delta_1$支配了$\delta_2$ 
* 如果不等关系对于某些$\theta$来说严格成立，就说这种支配关系是严格的。
* 如果一个估计器不被另外的估计器所支配，就说这个估计器是可容许的 (Admissible)。



##### **定理 6.3.3 （Admissibility is not enough）**

> 设有正态分布$X\sim N(\theta,1)$，在平方误差下对$\theta$进行估计。设$\delta_1(x)=\theta_0$是一个独立于数据的常量，则这是一个可容许估计器

**证明：**

用反证法，

* 假设结论不成立，存在另外一个估计器$\delta_2$有更小风险，所以有：

  $$R(\theta^*,\delta_2)\le R(\theta^*,\delta_1)$$ 

  对于某些$\theta^*$，以上不等关系严格成立

* 设真实参数为$\theta^*=\theta_0$，则$R(\theta^*,\delta_1)=0$，并且有：

  $R(\theta^*,\delta_2)=\int (\delta_2(x)-\theta_0)^2p(x\|\theta_0)dx$

* 由于对于所有的$\theta^*$，都有$0\le R(\theta^*,\delta_2)\le R(\theta^*,\delta_1)$

  而$R(\theta_0,\delta_1)=0$，所以有$R(\theta_0,\delta_2)=0,\delta_2(x)=\theta_0=\delta_1(x)$

因此，$\delta_2$只有和$\delta_1$相等的情况下才能避免在某一点$\theta_0$处有更高风险

即不能有其他的估计器$\delta_2$能严格提供更低的风险，所以$\delta_1$是可容许的



### 6.4 估计器的理想性质

#### 6.4.1 连续估计器

**定义**

> 随着取样规模趋近于无穷大，最终能够恢复出生成数据的真实参数的估计器，也就是随着$\|D\|\rightarrow \infin$，$\hat\theta(D)\rightarrow \theta^*$

最大似然估计就是一个连续估计器，直观理解就是因为将似然函数最大化其实就等价于将散度 $KL(p(*\|\theta^*)\|\|p(*\|\hat\theta))$ 最小化,其中的$p(*\|\theta^*)$是真实分布，而$p(*\|\hat\theta)$是估计的，很明显当且仅当$\hat\theta=\theta^*$的时候才有0散度。



#### 6.4.2 无偏估计器

估计器的偏差定义如下：

> $\text{bias}(\hat\theta( \cdot )) =\mathbb{E}_{p(D\|\theta_*)} [\hat\theta(D)-\theta_* ]   $

其中，$\theta_{*}$是真实的参数值；如果偏差为0，则称该估计器无偏差，即意味着取样分布的中心正好是真实参数。

**Example**

* 对高斯分布均值的最大似然估计就是无偏差的：

  $\text{bias}(\hat\mu)  =\mathrm{E}[\bar x]-\mu= =\mathrm{E}[\frac{1}{N{\sum ^N_{i=1}x_i}}] -\mu =\frac{N\mu}{N}-\mu=0$

  

* 对高斯分布方差的最大似然估计 $\hat \sigma ^2$ 不是对 $\sigma^2$ 的无偏估计

  $\mathbb{E} [\hat\sigma^2]=\frac{N-1}{N}\sigma^2$

  

* 对高斯分布方差的无偏估计器

  $\hat\sigma^2_{N-1}=\frac{N}{N-1}\hat\sigma^2=\frac{1}{N-1}\sum^N_{i=1}(x_i-\bar x)^2$

  > $\mathbb{E} [\hat\sigma^2_{N-1}]=\mathbb{E} [\frac{N}{N-1}\sigma^2]=\frac{N}{N-1}\frac{N-1}{N}\sigma^2=\sigma^2  $



#### 6.4.3 最小方差估计器

##### **定理 6.4.1（克莱默-饶不等式）**

> 设 $X_1,..,X_n \sim p(X\|\theta_0)$，而$\hat\theta=\hat\theta(x_1,..,x_n)$是一个对参数$\theta_0$的无偏估计器。然后在对$p(X\|\theta_0)$的各种平滑假设下，有：
>
> $\text{var} [\hat\theta]\ge \frac{1}{nI(\theta_0)}$
>
> 其中的 $I(\theta_0)$ 是费舍信息矩阵

**NOTE**：克莱默-饶下界为任意的无偏估计器的方差提供了下界，最大似然估计能达到克莱默-饶下界，因此在所有无偏估计器中拥有渐进的最小方差，所以说最大似然估计是渐进最优。



#### 6.4.4 偏差-方差权衡 

设 $\hat\theta =\hat \theta(D)$表示这个估计，然后 $\bar \theta =\mathrm{E}[\hat\theta]$ 表示的是估计的期望值，所有期望和方差都是关于真实分布$p(D\|\theta^*)$，均方误差的分解如下：
$$
\begin{aligned}
\mathrm{E} [(\hat\theta-\theta^*)^2 ]&=\mathrm{E} [[(\hat\theta-\bar \theta)+(\bar\theta-\theta^*) ]^2]  \\
&=\mathrm{E} [(\hat\theta-\bar \theta)^2] +2(\bar\theta-\theta^*)\mathrm{E}[\hat\theta-\bar\theta]+(\bar\theta-\theta^*)^2  \\
&=\mathrm{E}[(\hat\theta-\bar\theta)^2]+(\bar\theta-\theta^*)^2   \\
&=\text{var}[\hat\theta]+ \text{bias}^2(\hat\theta)   \\
\end{aligned}
$$
即，$MSE = variance + bias^2$，这就是偏差 - 方差之间的权衡。这就意味着假设我们的目标是要最小化平方误差，那么选择一个有偏差估计器也可能是可取的（只要能够降低方差）



### 6.5 经验风险最小化 (Empirical risk minimization)

**Problem**：

频率论决策方法难以避免的一个基本问题就是不能计算出风险函数，因为要知道真实数据分布才行。(作为对比，贝叶斯后验期望损失就总能计算出来，因为条件是在数据上的，而不是真实参数$\theta^*$）

**Solution**：

预测可观测量，而不是估计隐藏变量或者隐藏参数，即不寻找 $L(\theta,\delta(D))$ 形式的损失函数（其中的$\theta$是未知的真实参数，而$\delta(D)$是估计器），而是寻找 $L(y,\delta(x))$ 形式的损失函数，其中的 $y$ 是未知的真实响应变量，而$\delta(x)$是对给定的输入特征x做出的预测



此时，频率论的风险函数为：
$$
R(p_*,\delta) \overset{\triangle}{=} \mathrm{E}_{(x,y)\sim p_*}[L(y,\delta(x)]=\sum_x\sum_yL(y,\delta(x))p_*(x,y)
$$
上式中的 $p_*$ 表示的是真实的自然分布，该分布是未知的，但是可以使用一个经验分布来近似，这个经验分布是通过训练集数据来获得的：
$$
p_*(x,y)\approx p_{emp}(x,y) \overset{\triangle}{=}\frac{1}{N}\sum^N_{i=1}\delta_{x_i}(x)\delta_{y_i}(y)
$$
则经验风险定义如下：
$$
R_{emp}(D,D) \overset{\triangle}{=} R(P_{emp},\delta) =\frac{1}{N}\sum^N_{i=1}L(y_i,\delta(x_i))
$$

* 在 0-1 损失函数的情况下，上面的 $L(y,\delta(x))= I(y\ne \delta(x))$，经验风险变成了误分类率

* .在平方误差损失函数的情况下，上面的 $L(y,\delta(x))= (y-\delta(x))^2$​，经验风险就变成了均方误差，然后将经验风险最小化 (ERM）定义为找到一个能使经验风险最小化的决策过程（通常都是分类规则）：
  $$
  \delta_{ERM}(D)=\arg\min_{\delta}R_{emp}(D,\delta)
  $$

* 在无监督学习的情况下，可以去掉所有带 $x$ 的项，然后将 $L(y,\delta(x))$ 替换成 $L(x,\delta(x))$，例如，设$L(x,\delta(x))=\|\|x-\delta(x)\|\|^2_2$，则衡量的是重建误差。然后 $\delta(x)=decode(encode(x))$ 来定义决策规则，最终经验风险的形式定义如下：
  $$
  R_{emp}(D,\delta) =\frac{1}{N}\sum^N_{i=1} L(x_i,\delta(x_i))
  $$

#### 6.5.1 规范化风险最小化 (Regularized risk minimization)

如果我们关于 “自然分布” 的先验是完全等于经验分布，则经验风险等于贝叶斯风险：
$$
\mathbb{E}\left[R(p_*,\delta)|p_*=p_{emp}\right]=R_{emp}(D,\delta)
$$
因此，最小化经验风险可能会导致过拟合，所以通常都得为目标函数增加一个复杂度惩罚函数：
$$
R' (D,\delta)=  R_{emp}(D,\delta)+\lambda C(\delta)   
$$
上式中的 $C(\delta)$衡量的是预测函数 $\delta(x)$ 的复杂度，而 $\lambda$ 控制的是复杂度惩罚的程度，该方法称为规范风险最小化（RRM）。

**NOTE**：如果损失函数是对数似然函数的负数，那么规范化项就是负的对数先验，这也就等价于最大后验估计

规范化风险最小化有两个关键问题：

* 如何衡量复杂度
  * 对于线性模型来说，可以用其自由度定义成复杂度
  * 对于更多的通用模型,可以使用VC维度
* 如何挑选 $\lambda$



#### 6.5.2 结构风险最小化

规范化风险最小化原则表明，对于给定的复杂度惩罚函数，可以使用下面的公式来拟合模型：
$$
 \hat\delta_{\lambda}=\underset{\delta}{\text{argmin}}[R_{emp}(D,\delta)+\lambda C(\delta)]    
$$

* 如何选择 $\lambda$ 

  * 不能使用训练集，因为这会低估真实风险，也就是所谓的训练误差优化

  * 使用结构风险最小化原则：$\hat\lambda =\underset{\lambda}{\arg\min} \hat R(\hat \delta _{\lambda})$

    上式中的 $\hat R(\delta)$ 是对风险的估计，有两种广泛应用的估计：交叉验证以及风险理论上界约束



#### 6.5.3 使用交叉验证估计风险函数

可以利用一个验证集来估计某个估计器的风险，如果没有单独的验证集，可以使用交叉验证

交叉验证定义如下：

* 假设训练集中有 $N = \|D\|$ 个数据，将第 k 份数据表达为$D_k$，而其他的所有数据就表示为$D_{-k}$（在分层验证中，如果类标签是离散的，就选择让每份数据规模都基本相等）

* 设 ${F}$ 是一个学习算法或者拟合函数，使用数据集 $D$ 和模型索引 $m$（可以是离散索引，比如多项式指数，也可以是连续的，比如规范化强度等等），返回的是参数变量：
  $$
  \hat\theta_m =F(D,m)
  $$

* 最后，设P是一个预测函数，接受一个输入特征和一个参数向量，然后返回一个预测：
  $$
  \hat y=P(x,\hat \theta)=f(x,\hat \theta)
  $$
  这样就形成了一个拟合-预测循环：

  $$
  f_m(x,D)=P(x,F(D,m))
  $$

对$f_m$的风险函数的K折交叉验证估计的定义为:

$$
R(m,D,K)\overset{\triangle}{=} \frac{1}{N}\sum^N_{k=1}\sum_{i\in D_k}L(y_i, P(x_i,F(D_{-k},m)))
$$
然后就可以对每一份数据都调用运行一次拟合算法。



设 $f^k_m(x)=P(x,F(D_{-k},m))$ 是在第 k 折时要在除去测试集之外的所有数据上进行训练的函数，然后就可以把交叉验证估计改写成下面的形式：
$$
R(m,D,K)=\frac{1}{N}\sum^N_{k=1}\sum_{i\in D_k}L(y_i,f^{k}_m(x_i))=\frac{1}{N}\sum^N_{i=1}L(y_i,f^{k(i)}_m(x_i)) 
$$
其中，$k(i)$是所用的验证集所在折数，此时 $x_i$ 将被用作测试数据，即使用一个不包含$x_i$的数据训练出的模型来预测$y_i$



**留一交叉验证 (leave one out cross validation，LOOCV)**

如果 $K=N$，这个方法就变成了留一交叉验证 (leave one out cross validation，LOOCV)，此时估计的风险就变成了
$$
R(m,D,N)=\frac{1}{N}\sum^N_{i=1}L(y_i,f^{i}_m(x_i))
$$
上式中的$f^{i}_m(x_i)=P(x,F(D_{-i},m))$，这需要对模型进行N次拟合，其中$f^{i}_m$ 使用的是除去第 i 个训练样本外的所有数据



**通用交叉验证 (generalized cross validation，GCV)**

> 有的模型分类和损失函数（比如线性模型和平方损失函数）可以只拟合一次，然后以解析方式获得去除掉第 i 个训练样本的效果





#### 6.5.4 使用统计学习理论的风险上界 (Upper bounding the risk using statistical learning theory)*

​	略



#### 6.5.5 代理损失函数 (Surrogate loss functions)

**Problem**：在经验误差最小化(ERM) / 规范误差最小化(RRM)框架中最小化损失函数并不总是很简单，例如，0-1风险函数是非光滑的，很难去优化

**Solution**：用最大似然估计替代，因为对数似然函数是个光滑凸函数，是 0-1风险函数的上界



考虑二项逻辑回归，设$y_i\in \{-1, \ +1\}$，并且决策函数计算的是对数比值：
$$
f(x_i)=\log \frac{p(y=1|x_i,w)}{p(y=-1|x_i,w)}=w^Tx_i=\eta_i
$$
则对应的输出标签上的概率分布为：
$$
p(y_i|x_i,w)= sigm(y_i,\eta_i)
$$
对数损失函数定义为
$$
L_{nll}(y,\eta)=-\log p(y|x,w)=\log(1+e^{-y\eta})
$$
显然，最小化平均对数损失函数就等价于最大化似然函数

现在考虑计算最大概率标签，如果 $\eta_i < 0$，则 $\hat y = -1$；如果 $\eta_i \geq 0$，则 $\hat y = +1$，此时函数的 0-1 损失变成如下形式：
$$
L_{01}(y,\eta)= \mathbb{I}(y\ne \hat y)= \mathbb{I}(y\eta<0)
$$
 ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051735870.png)

如上图所示，对数损失函数显然是 0-1 损失函数的上界，即对数损失函数是代理损失函数的一个例子，另一个例子是铰链损失函数 (hinge loss)：
$$
L_{\text{hinge}}(y,\eta)=\max(0,1-y\eta)
$$
**NOTE**：代理损失函数通常是选凸上界的函数，因为凸函数容易最小化





### 6.6 频率论统计学的缺陷 (Pathologies of frequentist statistics)*

​	略





## C7. Linear regression

### 7.1 模型规范

* 线性回归的形式如下：

$$
p(y|x, \theta)=N(y|w^Tx,\sigma^2)
$$

* 线性回归可以通过将 $x$ 替换成输入特征的非线性函数 $\phi(x)$ 来对非线性关系进行建模，即将形式变成：
  $$
  p(y|x, \theta)=N(y|w^T\phi (x), \sigma^2)
  $$
  这就是**基函数扩展**，此时的模型依然是以 $w$ 为参数，即仍然为线性模型

  例子：多项式基函数

  模型中函数形式为
  $$
  \phi(x)=[1,x,x^2,...,x^d]
  $$
  增加维数 d 可以建立更为复杂的函数

* 对于多输入的模型，也可以使用线性回归

  例如，将温度作为地理位置的函数来建模：

  图（a）：$\mathbb{E}[y\|x]=w_0+w_1x_1+w_2x_2$

  图（b）：$\mathbb{E}[y\|x]=w_0+w_1x_1+w_2x_2+w_3x_1^2+w_4x_2^2$

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051735160.png)



### 7.2 最大似然估计（最小二乘法） 

最大似然估计是估计统计模型参数的常用方法，其定义如下：
$$
\hat\theta \overset{\triangle}{=} \underset{\theta}{\arg \max} \log p(D|\theta)
$$
通常假设训练样本都是独立同分布的，则可以写出下面的对数似然函数：
$$
l(\theta) \overset{\triangle}{=}\log p(D|\theta) =\sum \limits ^N_{i=1}\log p(y_i|x_i,\theta)
$$
目标：最大化对数似然函数，或者等价地最小化负数对数似然函数（negative log likelihood, NLL）：
$$
NLL(\theta)\overset{\triangle}{=} -\sum^N_{i=1}\log p(y_i|x_i,\theta)
$$


对线性回归模型使用最大似然估计法，在 $l(\theta)$ 的公式中加入高斯分布的概率密度函数，即可得到以下形式的对数似然函数：
$$
\begin{aligned}
l(\theta)&=  \sum^N_{i=1}\log \left[(\frac{1}{2\pi\sigma^2})^{\frac{1}{2}} \exp (-\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2)  \right ]\\
&= \frac{-1}{2\sigma^2}RSS(w)-\frac{N}{2}\log(2\pi\sigma^2)\\
\end{aligned}
$$
其中，RSS（residual sum of squares）为残差平方和，其定义如下：
$$
RSS(w)\overset{\triangle}{=} \sum^N_{i=1}(y_i-w^Tx_i)^2
$$
RSS 也叫做平方误差和（sum of squared errors，SSE），SSE / N则被称为均方误差（mean squared error，MSE）。RSS也可以写成残差向量的二阶范数的pSS 也叫做平方误差和（sum of squared errors，SSE），SSE / N则被称为均方误差（mean squared error，MSE）。RSS也可以写成残差向量的二阶范数的平方：
$$
RSS(w)=||\epsilon||^2_2=\sum^N_{i=1}\epsilon_i^2
$$
上式中的 $\epsilon_i = (y_i - w^Tx_i)^2$。显然，$w$ 的最大似然估计就是使残差平方和最小的 $w$。



#### 7.2.1 最大似然估计的派生 （Derivation of the MLE）

* 将目标函数（负对数似然函数）重写为以下形式：
  $$
  NLL(w)=\frac{1}{2}(y-Xw)^T(y-Xw)=\frac{1}{2}w^T(X^TX)w-w^T(X^Ty)
  $$
  上式中
  $$
  X^TX=\sum^N_{i=1}x_ix_u^T=\sum^N_{i=1}\begin{pmatrix} x_{i,1}^2&... x_{i,1}x_{i,D}\\&& ...&\\  x_{i,D}x_{i,1} &... & x_{i,D}^2 \end{pmatrix}
  $$
  
  是平方和矩阵，另外一项
  $$
  X^Ty = \sum \limits ^{N}_{i=1} x_i y_i
  $$
  
* 梯度函数为：
  $$
  g(w)=[X^TXw-X^Ty]=\sum^N_{i=1} x_i(w^Tx_i-y_i)
  $$

  > 使用了等式 4.10 中的结论
  >
  > <img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051736600.png" style="zoom:67%;" />



* 令梯度为零，则有：
  $$
  X^TXw=X^Ty
  $$
  这就是正规方程（normal equation），这个线性方程组对应的解 $\hat w$ 叫做常规最小二乘解（ordinary least squares solution，OLS solution）：
  $$
  \hat w_{OLS}=(X^TX)^{-1}X^Ty
  $$
  

#### 7.2.2 几何解释

* 假设 $N > D$，即样本比特征数目多，$X$ 的列向量定义了嵌入在 N 维中的一个 D 维线性子空间，设 $\tilde x_j$ 是 $X$ 的第 $j$ 列，则有 $\tilde x_j, y \in \mathbb R^N$。

  例如，设在 $D =2$ 维度上有 $N = 3$ 个样本


$$
X=\begin{pmatrix}1&2 \\ 1 &-2\\1 &2 \end{pmatrix},y=\begin{pmatrix}8.8957\\0.6130\\1.7761\end{pmatrix}
$$

* 目标是在 $D$ 维线性子空间中找到一个尽可能靠近 $y$ 的向量 $\hat y \in \mathbb R^N$，即要寻找：
  $$
  \underset{\hat y \in span(\{ \tilde x_1,...,\tilde x_D \})}{\arg\min} ||y-\hat y||_2
  $$
  

  由于 $\hat y \in span(X)$，因此会存在某个权重向量 $w$ 使得：
  $$
  \hat y= w_1\tilde x_1+...+w_D\tilde x_D=Xw
  $$
  
* 要最小化残差的范数 $y-\hat y$，就需要让残差向量和 $X$ 的每一列相正交,，即对于$j=1:D$ 有 $\tilde x ^T_j (y-\hat y) =0$，因此有:
  $$
  \tilde x_j^T(y-\hat y)=0  \implies X^T(y-Xw)=0\implies w=(X^TX)^{-1}X^Ty
  $$
  所以 $y$ 的投影值为：
  $$
  \hat y=X\hat w= X(X^TX)^{-1}X^Ty
  $$
  这对应着 $y$ 在 $X$ 的列空间中的正交投影，投影矩阵 $P \overset \triangle{=} X(X^TX)^{-1}X^T$ 被称为帽子矩阵（hat matrix）



#### 7.2.3 凸性质

负对数似然函数为凸函数

* **凸集**：

  设一个集合 $S$，如果对于任意的 $\theta, \theta' \in S$，有
  $$
  \lambda\theta + (1-\lambda)\theta' \in S, \ \forall \lambda \in [0, 1]
  $$
  则集合 $S$ 为凸集

<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051736337.png" style="zoom:67%;" />

* **凸函数**

  * 一个函数的 **epigraph** （函数上方的全部点组成的集合）定义了一个凸集合，则称这个函数 $f(\theta)$ 为凸函数

  * 如果定义在一个凸集合 $S$ 上的函数$f(\theta)$ 满足对任意的 $\theta, \theta' \in S$，以及任意的 $0 \leq \lambda \leq 1$ 都有
    $$
    f(\lambda \theta + (1-\lambda)\theta') \leq \lambda f(\theta) + (1-\lambda)f(\theta')
    $$
    （凸组合的函数值 $\leq$ 函数值的凸组合）

    * 如果不等式严格成立，则这个函数是严格凸函数，如果函数$-f(\theta)$是凸函数，则函数$f(\theta)$ 是凹函数
    * 标量凸函数包括$\theta^2,e^\theta,\theta\log\theta (\theta>0)$；标量凹函数包括$\log(\theta),\sqrt\theta$
    * 严格凸函数会存在全局最小值点 $\theta ^ *$

    <img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051736157.png" style="zoom:67%;" />

  * 对于一元函数 $f(\theta)$，二阶导数 $\geq 0$ $\Longrightarrow$ $f(\theta)$ 为凸函数，若不等式严格成立，则为严格凸函数；对于多元函数 $f$，$f$ 为凸函数当且仅当海森矩阵为半正定矩阵（正定 $\Longrightarrow$ 严格凸）





### 7.3 健壮线性回归*

​	略



### 7.4 岭回归（Ridge regression）

最大似然估计中一大问题就是过拟合，本节将使用高斯先验的最大后验估计来改善过拟合问题



#### 7.4.1 基本思想

最大似然估计过拟合的原因就是它选的参数都是最适合对训练数据建模的，但如果训练数据有噪音，这些参数就经常会形成非常复杂的函数（为了完美拟合所有数据，系数可能会有特别大的数值，但是这将会很不稳定）

* 使用一个零均值高斯先验可以让系数小一点，这样得到的就是更光滑的曲线：
  $$
  p(w)=\prod_j N(w_j|0,\tau^2)
  $$
  其中，$1/ \tau^2$控制了先验强度

  

* 对应的最大后验问题为：
  $$
  \underset{w}{\arg\max}\sum^N_{i=1}\log N(y_i|w_0+w^Tx_i,\sigma^2)+\sum^D_{j=1}\log(w_j|0,\tau^2)
  $$
  该问题等价于对下面这个式子求最小值：
  $$
  J(w)=\frac{1}{N}\sum^N_{i=1}(y_i-(w_0+w^Tx_i))^2+\lambda||w||^2_2
  $$
  其中的 $\lambda\overset{\triangle}{=} \sigma^2/\tau^2$，$\|\|w\|\|^2_2=\sum_j w^Tw$为平方二范数，上式中的第一项依然是均方误差/负对数似然函数，第二项为复杂度惩罚项，其中$\lambda \geq 0$

  

* 上述问题对应的解为：
  $$
  \hat w_{ridge}=(\lambda I_D+X^TX)^{-}X^Ty
  $$

上述方法即为岭回归（ridge regression），也叫惩罚最小二乘法；通常情况下将使用高斯分布先验来使参数变小的方法叫做 $l_2$ 规范化 or 权重衰减；通过对权重大小之和进行惩罚，能确保函数尽量简单。

**NOTE**：偏移项$w_0$并不是规范化的，因为这只影响函数的高度，而不影响其复杂性



#### 7.4.2 数值稳定计算*

​	略

#### 7.4.3 和主成分分析(PCA)的联系*

​	略

#### 7.4.4 大规模数据的规范化效应

规范化是避免过拟合的最常用方法，而另外一种有效的方法，就是使用大规模数据。直观来看就是训练用的数据规模更多，进行学习的效果就能越好，所以我们期望随着数据规模N增大，测试误差就逐渐降低到某个定值

测试集上误差的形态有两方面决定：

* 生成过程中的内在变异性导致的对于所有模型都会出现的无法降低的部分（也叫作噪声基底，noise floor）；
* 另一个依赖于生成过程（真实情况）和模型之间差异导致的部分（也叫作结构误差，structural error)。

对于任何足以捕获真实情况的模型（即有最小结构误差），测试误差都会随着样本规模增大即$N\rightarrow \infty$而趋向噪声基底



**近似误差**：对于有限规模的训练集而言，我们估计的参数和给定模型类别能进行估计的最佳参数之间总是会有一些差异

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051737589.png)



### 7.5 贝叶斯线性回归

虽然岭回归是计算点估计的有效方法，有时候还可能要对 $w$ 和 $\sigma^2$ 的全后验进行计算。为了简单起见，就假设噪声方差$\sigma^2$已知，只需要关注与计算$p(w\|D,\sigma^2)$。



#### 7.5.1 计算后验

在线性回归中，似然函数为：
$$
\begin{aligned}
p(y|X,w,\mu,\sigma^2)& =N(y|\mu+Xw,\sigma^2I_N)             \\
& \propto \exp(-\frac{1}{2\sigma^2}(y-\mu1_N-Xw)^T(y-\mu1_N-Xw))        \\
\end{aligned}
$$
其中，$\mu$是偏移项，如果输入值是中心化的，则对于每个 $j$ 都有 $\sum_ix_{ij}=0$，输出均值的正负概率相等。所以为 $\mu$ 假设一个不适当先验，形式为$p(\mu)\propto 1$，然后再整合起来就可以得到：
$$
p(y|X,w,\sigma^2)\propto \exp( -\frac{1}{2\sigma^2}||  y-\bar y1_N-Xw  ||^2_2 )
$$
其中，$\bar y =\frac{1}{N} \sum^N_{i=1} y_i$ 是输出的经验均值。

上面这个高斯似然函数的共轭先验还是高斯分布，可以表示为 $p(w)= N(w\|w_0,V_0)$，利用高斯分布的贝叶斯规则，可以得到下面的后验：
$$
\begin{aligned}
p(w|X,y,\sigma^2)& \propto N(w|w)0,V_0)N(y|Xw,\sigma^2I_N)=N(w|w_N,V_N)\\
W_N& = V_NV_0^{-1}w_0+\frac{1}{\sigma^2}V_NX^Ty \\
V_N^{-1}& = V_0^{-1}+\frac{1}{\sigma^2}X^TX  \\
V_N& = \sigma^2(\sigma^2V_0^{-1}+X^TX)^{-1} \\
\end{aligned}
$$
如果$w_0=0,V_0=\tau ^2I$，且定义$\lambda=\frac{\sigma^2}{\tau^2}$，那么后验均值就降低到了岭估计（因为高斯分布的均值和众数相等）

>高斯分布的贝叶斯规则
>
><img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051737587.png" style="zoom:67%;" />





#### 7.5.2 计算后验预测

> It’s tough to make predictions, especially about the future. — Yogi Berra

利用以下等式
$$
p(y)=N(y|A\mu_x+b,\Sigma_y+A\Sigma_x A^T)
$$
可以发现对于测试点 $x$ 的后验预测分布也是一个高斯分布：
$$
\begin{aligned}
p(y|x,D,\sigma^2)& = \int N(y|x^Tw,\sigma^2)N(w|w_N,V_N)d w  \\
&= N(y|w^T_Nx,\sigma^2_N(x)) \\
\sigma^2_N(x)&= \sigma^2+x^TV_Nx  \\
\end{aligned}
$$
该预测分布中的方差 $\sigma^2_N(x)$ 取决于两个项：观测噪声的方差 $\sigma^2$ 和 参数方差 $V_N$。后面这一项表示为观测方差，取决于测试点 $x$ 和训练数据集 $D$ 之间的距离。此外，误差范围会随着远离训练样本中的点而增大，表示着不确定性的增加。对比之下，插值估计就有固定的误差范围，因为：
$$
p(y|x,D,\sigma^2)\approx \int N(y|x^Tw,\sigma^2)\delta_{\hat w}(w)d w=p(y|x,\hat w,\sigma^2 )
$$


#### 7.5.3 $\sigma^2$未知的情况下用贝叶斯推断 *

​	略



#### 7.5.4 线性回归的经验贝叶斯方法（证据程序，evidence procedure）

**证据程序（evidence procedure）**

挑选能够将边缘似然函数最大化的 $\eta=(\alpha,\lambda)$，其中 $\lambda=1/\sigma^2$ 是观测噪声的精度，而$\alpha$是先验精度，先验为 $p(w)=N(w\|0,\alpha^{-1}I)$

* 证据程序可以作为交叉验证的一个替代方法

* 对比不同类别模型的时候，证据程序提供了对证据的一个很好的估计：
  $$
  \begin{aligned}
  p(D|m)&= \int \int p(D|w,m)p(w|m,\eta)p(\eta|m)dw d\eta \\
  &\approx \max_\eta \int p(D|w,m)p(w|m,\eta)p(\eta|m)dw \\
  \end{aligned}
  $$
  





## C8. Logistic regression

### 8.1 概论

构建概率分类器的两种方法：

* 生成模型：建立形式为 $p(y,x)$ 的联合模型，然后以 $x$ 为条件，推导 $p(y\|x)$
* 判别模型：直接以 $p(y\|x)$ 的形式去拟合一个模型

本章内容主要围绕需要假设有一些参数为线性的判别模型进行阐述



### 8.2 模型规范

逻辑回归对应的是下面这种二值化分类模型：
$$
p(y|x, w) = Ber(y | sigm(w^Tx))
$$


### 8.3 模型拟合

#### 8.3.1 最大似然估计

逻辑回归的负对数似然函数为
$$
\begin{aligned}
NLL(w)&= -\sum^N_{i=1}\log \left[\mu_i^{\mathbb{I}(y_i=1)}\times (1-\mu_i)^{\mathbb{I}(y_i=0)}\right] \\
&=  -\sum^N_{i=1}\log \left[y_i\log \mu_i+(1-y_i)\log(1-\mu_i)\right]   \\
\end{aligned}
$$
这也被称为交叉熵误差函数（cross-entropy error function）

* $y_i$ 表示样本 $i$ 的 label，$y_i \in \{0, 1\}$

* $\mu_i$ 表示样本 $i$ 预测为正类的概率

  

假设 $\tilde y \in \{-1, +1 \}$，而不是 $y_i \in \{0, 1\}$，并且 $p(y = -1) = \frac{1}{1+ \exp{(-w^Tx)}}$，$p(y = 1) = \frac{1}{1+ \exp{(+w^Tx)}}$，则有：
$$
NLL(w)=\sum^N_{i=1}\log(1+\exp(-\tilde y_i w^Tx_i))
$$
与线性回归不同，在逻辑回归里面，我们无法以闭合形式写出最大似然估计，所以需要使用优化算法来进行计算，因此需要对梯度和海森矩阵进行推导：
$$
\begin{aligned}
g &=\frac{d}{dw}f(w)=\sum_i*\mu_i-y_i)x_i=X^T(\mu-y)  \\
H &= \frac{d}{dw}g(w)^T=\sum_i(\nabla_w\mu_i)x_i^T=\sum_i\mu_i(1-\mu_i)x_xx_i^T \\
&= X^TSX  \\
\end{aligned}
$$
其中，$S \overset \triangle = diag(\mu_i(1-\mu_i))$，海森矩阵 $H$ 是正定的，因此，负对数似然函数是凸函数，有唯一的全局最小值。



#### 8.3.2 梯度下降

无约束优化问题最简单的算法就是梯度下降（gradient descent），也叫最陡下降（steepest descent），其形式如下：
$$
\theta_{k+1} = \theta_k - \eta_k g_k
$$
其中，$\eta_k$ 为步长或者学习率



梯度下降法的主要问题在于如何设置步长：

* 若使用固定步长
  * 步长太小，收敛速度太慢
  * 步长太大，可能最终无法收敛

* 需要想办法保证无论起点在哪里最终都能收敛到局部最优值（这个性质叫做全局收敛性，与收敛到全局最优值进行区别！）



通过泰勒定理（带拉格朗日余项的泰勒公式）可以得到：
$$
f(\theta+\eta d)\approx f(\theta)+\eta g^Td
$$
其中，$d$ 为下降方向，所以如果 $\eta$ 足够小，则有 $f(\theta + \eta d) < f(\theta)$（因为梯度会是负值的）

步长太小会导致收敛速度过慢，因此需要选择一个能够最小化下面这个项的步长 $\eta$：
$$
\phi(\eta) = f(\theta_k + \eta_k d_k)
$$
这就叫线性最小化或者线性搜索



**Problems**

线性搜索得到的梯度下降路径会有一种扭折行为（zig-zag behavior）：

梯度下降过程中一次特定的线性搜索中想要满足 $\eta_k = \underset{\eta >0}{\arg\min} \phi(\eta)$，则优化的一个必要条件是导数为零，即 $\phi'(\eta) = 0$，由链式法则可以得到：
$$
\phi'(\eta) = d^Tg
$$
其中，$g = f'(\theta + \eta d)$ 是当前步骤结束时的梯度，因此 $\phi'(\eta) = 0$ 意味着：

* 要么 $g = 0$，即已经到达一个稳定点
* 要么 $g \perp d$，即这一步结束时所在位置的局部梯度与搜索方向相互垂直，因此连续起来搜索方向就是正交的



**Solutions**

* 降低这种扭折效应的一种简单的启发式方法就是增加一个动量项 $\theta_k - \theta_{k-1}$：
  $$
  \theta_{k+1}=\theta_k-\eta_kg_k+\mu_k(\theta_k-\theta_{k-1})
  $$
  上式中的 $0 \leq \mu_k \le 1$ 控制了动量项的重要程度。在优化领域中，这个方法叫做重球法。



* 另外一种最小化扭折行为的方法是使用共轭梯度，这是 $f(\theta)=\theta^TA\theta$ 形式的二次目标所选择的方法，这种二次目标会在解线性系统时出现



#### 8.3.3 牛顿法

**最小化一个严格凸函数的牛顿法** 

[牛顿法推导过程](https://zhuanlan.zhihu.com/p/3354436 )

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051737777.png)

> **二阶优化方法：**考虑空间曲率
>
> **牛顿法：**不考虑空间曲率

牛顿法是一个迭代算法，其中包含以下形式的更新步骤：
$$
\theta_{k+1}=\theta_k-\eta_kH_k^{-1}g_k
$$
在最简单的形式下，牛顿法需要海森矩阵$H_k$为正定矩阵，这保证了目标函数是严格凸函数。否则，目标函数非凸函数，则海森矩阵$H_k$就可能不正定了，所以$d_k=-H_k^{-1}g_k$就可能不是一个下降方向



#### 8.3.4 迭代重加权最小二乘法（Iteratively reweighted least squares，IRLS）

将牛顿法应用到二值化逻辑回归中来求最大似然估计，在这个模型中第 $k+1$ 次迭代中牛顿法更新如下所示：（因为海森矩阵是确定的，所以设 $\eta_k = 1$）
$$
\begin{aligned}
w_{k+1}&=  w_k-H^{-1}g_k &\text{(8.18)}\\
&= w_k+(X^TS_kX)^{-1}X^T(y-\mu_k)   &\text{(8.19)}\\
&= (X^TS_kX)^{-1}[(X^TS_kX)w_k+X^T(y-\mu_k)]  &\text{(8.20)}\\
&= (X^TS_kX)^{-1}X^T[S_kXw_k+y-\mu_k]  &\text{(8.21)}\\
&= (X^TS_kX)^{-1}X^TS_kz_k  &\text{(8.22)}\\
\end{aligned}
$$
定义工作响应函数如下所示：
$$
z_k\overset{\triangle}{=} Xw_k+S_k^{-1}(y-\mu_k)
$$


等式8.22就是一个加权最小二乘问题，是要对下面的项最小化：
$$
z_{ki}=w_k^Tx_i+\frac{y_i-\mu_{ki}}{\mu_{ki}(1-\mu_{ki})}
$$
由于$S_k$是一个对角矩阵，所以可以把目标函数重写为成分形式（对每个$i=1:N$）：
$$
z_{ki}=w_k^Tx_i+\frac{y_i-\mu_{ki}}{\mu_{ki}(1-\mu_{ki})}
$$
这个算法就叫做迭代重加权最小二乘法，因为每次迭代都解一次加权最小二乘法，其中的权重矩阵$S_k$在每次迭代都会变化

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051739777.png)



#### 8.3.5 拟牛顿法

牛顿法以及二阶优化算法都需要计算海森矩阵 $H$，然而通常情况下海森矩阵的运算开销会较高。而拟牛顿法则是以迭代方式使用从每一步的梯度向量中学到的信息来构建对海森矩阵的估计，最常用的方法就是 [BFGS](https://www.cnblogs.com/ooon/p/5729982.html) 方法，该方法使用下面定义的 $B_k \approx H_k$ 来对海森矩阵进行估计：
$$
\begin{aligned}
B_{k+1}& =B_k+\frac{y_ky_k^T}{y_k^Ts_k}-\frac{(B_ks_k)(B_ks_k)^T}{s_k^TB_ks_k}\\
s_k& = \theta_k-\theta_{k-1} \\
y_k& = g_k-g_{k-1}  \\
\end{aligned}
$$
这是对矩阵的二阶更新，确保了矩阵保持正定（在每个步长的特定限制下），通常使用一个对角线估计来作为算法的初始化：$B_0 = I$。所以 BFGS 方法可以看作是对海森矩阵使用对角线加上低阶估计的方法。

此外，BFGS方法也可以对海森矩阵的逆矩阵进行近似，通过迭代更新 $C_l \approx H_k ^{-1}$，如下所示：
$$
C_{k+1}=(I-\frac{s_ks_k^T}{y_k^Ts_k})C_k(I- \frac{y_ks_k^T}{y_k^Ts_k})+\frac{s_ks_k^T}{y_k^Ts_k}
$$

* 存储海森矩阵需要消耗 $O(D^2)$ 的存储空间，所以对于很大规模的问题，可以使用限制内存BFGS算法（limited memory BFGS，L-BFGS），其中的 $H_k$ 或者 $H_k^{-1}$ 都是用对角矩阵加上低阶矩阵来近似的。具体来说就是积 $H_k^{-1}g_k$ 可以通过一系列的 $s_k$ 和 $y_k$ 的内积来得到，只使用m个最近的（$s_k,y_k$）对，忽略掉更早的信息，这样存储上就只需要$O(mD)$规模的空间了。



#### 8.3.6 $l_2$规范化

* 相比之下，我们更倾向于选岭回归而不是线性回归

* 类似地，对于逻辑回归我们更应该选最大后验估计而不是计算最大似然估计。

  > 实际上即便数据规模很大，在分类背景下规范化还是很重要的。假设数据是线性可分的，这时候最大似然估计就可以通过 $\|\|m\|\|\rightarrow \infty$ 来得到，对应的就是一个无穷陡峭的S形函数 (sigmoid function) $I(w^Tx>w_0)$，这也称为一个线性阈值单元，这将训练数据集赋予了最大规模概率质量，不过这样一来求解就很脆弱而且不好泛化。

所以需要和岭回归里面一样使用 $l_2$规范化，这样一来新的目标函数、梯度函数、海森矩阵就如下所示：
$$
\begin{aligned}
f'(w)&=NLL(w)+\lambda  w^Tw          \\
g'(w)&= g(w)+\lambda  w        \\
H'(w)&= H(w)+\lambda  I        \\
\end{aligned}
$$



#### 8.3.7 多类逻辑回归

**多类逻辑回归**也叫做最大**最大熵分类器**，其模型形式为
$$
p(y=c|x,W)=\frac{\exp(w_c^Tx)}{\sum^C_{c'=1}\exp(w_{c'}^Tx)}
$$
变体：**条件逻辑模型**（conditional logit model）

> 对每个数据案例的不同类别集进行归一化处理
>
> 可以用于对用户在不同组合的项目集合之间进行的选择进行建模



**Notation**

>设 $\mu_{ic}=p(y_i=c\|x_iW)=S(\eta_i)c$
>
>其中的 $\eta_iW^Tx_i$ 是一个$C\times 1$ 向量
>
>然后设 $y_{ic}=I(y_i=c)$是对 $y_i$ 的一种编码方式，使得 $y_i$ 成为二进制位向量，当且仅当 $y_i=c$ 的时候第 c 个位的值为1
>
>设$w_C=0$（保证可识别性），然后定义 $w=vec(W(:,1:C-1))$ 是一个$D\times (C-1)$ 维度的列向量

对数似然函数的形式如下：
$$
\begin{aligned}
l(W)&= \log\prod^N_{i=1}\prod^C_{c=1}\mu_{ic}^{y_{ic}}=\sum^N_{i=1}\sum^C_{c=1}y_{ic}\log \mu_{ic}   \\
&= \sum^N_{i=1}\left[(\sum^C_{c=1}y_{ic}w_c^Tx_i)-\log(\sum^C_{c'=1} \exp (w_{c'}^Tx_i) )\right]   \\
\end{aligned}
$$
负对数似然函数NLL：$f(w) = -l(w)$

* 针对负对数似然函数计算器梯度和海森矩阵，由于 $w$ 是分块结构，可以定义一个 $A\otimes B$ 表示在矩阵 $A$ 和矩阵 $B$ 之间的克罗内克积（kronecker product），如果 $A$ 是一个 $m\times n$ 矩阵，$B$是一个 $p\times q$ 矩阵，那么 $A\otimes B$ 就是一个 $mp\times nq$ 的分块矩阵：
  $$
  A\otimes B= \begin{bmatrix} a_{11}B &...& a_{1n}B\\...&...&...\\a_{m1}B&...&a_{mn}B \end{bmatrix}
  $$
  
* 梯度为：
  $$
  g(W)=\nabla f(w)=\sum^N_{i=1}(\mu_i-y_i)\otimes x_i
  $$
  其中，$y_i=(I(y_i=1),...,I(y_i=C-1))$ 和 $\mu_i(W)=[p(y_i=1\|x_i,W),...,p(y_i=C-1\|x_i,W)]$  都是长度为$C-1$的列向量

  例如，如果有特征维度D=3，类别数目C=3，$g(W)$ 的形式如下：
  $$
  g(W)=\sum_i\begin{pmatrix}(\mu_{i1}-y_{i1})x_{i1} \\(\mu_{i1}-y_{i1})x_{i2}\\(\mu_{i1}-y_{i1})x_{i3}\\(\mu_{i2}-y_{i2})x_{i1}\\(\mu_{i2}-y_{i2})x_{i2}\\(\mu_{i2}-y_{i2})x_{i3}\end{pmatrix}
  $$
  即，对于每个类 $c$，第 $c$ 列中权重的导数为 $\nabla_{w_c}f(W)=\sum_i(\mu_i{i}-y_{ic})x_i$



* 海森矩阵是一个大小为 $D(C-1)\times D(C-1)$ 的分块矩阵，其形式如下：
  $$
  H(W) = \nabla^2 f(w)=\sum^N_{i=1}(diag(\mu_i)-\mu_i\mu_i^T)\otimes (x_ix_i^T) 
  $$
  例如，有三个特征和三种类别，则 $H(W)$ 的形式为：
  $$
  \begin{aligned}
  H(W)&=\sum_i \begin{pmatrix} \mu_{i1}-\mu_{i1}^2 & -\mu_{i1}\mu_{i2}\\ -\mu_{i1}\mu_{i2}& \mu_{i2}-\mu_{i2}^2\end {pmatrix} \otimes \begin{pmatrix} x_{i1}x_{i1}&x_{i1}x_{i2}&x_{i1}x_{i3}\\x_{i2}x_{i1}&x_{i2}x_{i2}&x_{i2}x_{i3}\\x_{i3}x_{i1}&x_{i3}x_{i2}&x_{i3}x_{i3}  \end{pmatrix}               \\
  &=  \sum_i\begin{pmatrix} (\mu_{i1}-\mu_{i1}^2)X_i&-\mu_{i1}\mu_{i2}X_i\\-\mu_{i1}\mu_{i2}X_i&(\mu_{i2}-\mu_{i2}^2)X_i  \end{pmatrix}              \\
  \end{aligned}
  $$
  其中，$X_i=x_ix_i^T$

  即，分块矩阵中块$c,c'$部分为：
  $$
  H_{c,c'}(W)=\sum_i\mu_{ic}(\delta_{c,c'}-\mu_{i,c'})x_ix_i^T
  $$
  **这是一个正定矩阵，所以有唯一最大似然估计**

  

* 接下来考虑最小化下面这个式子：
  $$
  f'(W) \overset{\triangle}{=} -\log (D|w)-\log p(W)
  $$
  其中，$p(W)=\prod_c \N(w_C\|0,V_0)$

  新的目标函数、梯度函数和海森矩阵形式如下：
  $$
  \begin{aligned}
  f'(w)&= f(w)+\frac{1}{2}\sum_cw_cV_0^{-1}w_c  \\
  g'(w)&= g(W)+V_0^{-1}(\sum_cw_c)  \\
  H'(w)&= H(W)+I_C\otimes V_0^{-1}  \\
  \end{aligned}
  $$
  这样就可以传递给任意的基于梯度的优化器来找到最大后验估计



### 8.4 贝叶斯逻辑回归

> 对于逻辑回归来说没有合适的共轭先验，因此不能像线性回归一样计算在参数上的完整后验分布 $p(w\|D)$，本节阐述的是简单的近似方法



#### 8.4.1 拉普拉斯近似

本节将对以下的后验分布进行高斯近似：

> 假设 $\theta \in \mathbb{R} ^ D$，设
> $$
> p(\theta | D) = \frac{1}{Z} e^{-E(\theta)}
> $$

其中，$E(\theta)$ 为能量函数，等于未归一化后验的负对数，即
$$
E(\theta) = - \log p(\theta, D)
$$
$Z=p(D)$ 是归一化常数

在众数 $\theta ^{\star}$ 处（即最低能量状态）进行泰勒展开，则可以得到：
$$
E(\theta) \approx E(\theta^*)+(\theta-\theta^*)^Tg+\frac{1}{2}(\theta-\theta^*)^TH(\theta-\theta^*)
$$
其中，$g$ 是梯度，$H$ 是在众数位置能量函数的海森矩阵：
$$
g\overset{\triangle}{=} \nabla E(\theta)|_{\theta^*} \\H\overset{\triangle}{=} \frac{\partial^2E(\theta)}{\partial\theta\partial\theta^T}|_{\theta^*}
$$
由于 $\theta ^*$ 是众数，梯度项为零，因此有
$$
\begin{aligned}
\hat p(\theta|D)& \approx \frac{1}{Z}e^{-E(\theta^*)} \exp[-\frac{1}{2}(\theta-\theta^*)^T H(\theta-\theta^*)] &\text{(8.52)}\\
& = N(\theta|\theta^*,H^{-1})&\text{(8.53)}\\
Z=p(D)& \approx \int \hat p(\theta|D)d\theta = e^{-E(\theta^*)}(2\pi)^{D/2}|H|^{-\frac{1}{2}} &\text{(8.54)}\\
\end{aligned}
$$

* 等式 8.54 是对边缘似然函数的拉普拉斯近似（由多元高斯的归一化常数得出），所以等式 8.52 有时候也叫做对后验的拉普拉斯近似



#### 8.4.2 贝叶斯信息量 (Bayesian information criterio，BIC) 的推导

使用高斯近似来写出对数似然函数，并去掉不相关常数之后的形式如下：
$$
\log p(D)\approx \log(p(D|\theta^*)+\log p(\theta^*)-\frac{1}{2}\log|H|
$$

* 加在 $\log(p(D\|\theta^*)$ 后面的惩罚项也叫作奥卡姆因子，是对模型复杂程度的度量

* 如果使用均匀先验，即 $p(\theta)\propto 1$，就可以去掉第二项，然后把 $\theta^*$ 替换成最大似然估计 $\hat\theta$ ：

  > $$
  > \log p(D)\approx \log(p(D|\hat\theta)-\frac{1}{2}\log|H|
  > $$
  >
  > 

* 对式子中的第三项进行估计：

  已知 $H = \sum _{i=1}^{N} H_i$，其中 $H_i=\nabla\nabla \log p(D_i\|\theta)$，使用一个固定的矩阵 $\hat H$ 来近似每个 $H_i$ ，即可得到：
  $$
  \log|H|=\log|N\hat H| =\log (N^d|\hat H|)=D\log N+\log |\hat H|
  $$
  其中，$D=dim(\theta)$

  假设 $H$ 是满秩矩阵，则可以去掉 $\log \|\hat H\|$ 这一项（因为其独立于 $N$ ，所以可以被似然函数盖过去）

综上，贝叶斯信息量分数的形式如下：
$$
\log p(D)\approx p(D|\hat \theta)-\frac{D}{2}\log N
$$

#### 8.4.3 逻辑回归的高斯近似

对逻辑回归应用高斯近似：

使用一个高斯先验，其形式为 $p(w)=N(w\|0,V_0)$，则近似后验为：
$$
p(w|D)\approx N(w|\hat w, H^{-1})
$$
其中，$\hat w = \arg\min_w \mathrm{E}(w),\ \mathrm{E}(w) =-(\log p(D\|w)+\log p(w)), \ H=\nabla^2 \mathrm{E}(w)\|_{\hat w}$



#### 8.4.4 近似后验预测

后验预测分布的形式为：
$$
p(y|x,D)=\int p(y|x,w)p(w|D)dw
$$
这个积分直接计算会比较麻烦，最简单的近似就是差值估计，在二值化分类的情况下，其形式为：
$$
p(y=1|x,D)\approx p(y=1|x,\mathrm{E}[w])
$$
其中，$E[w]$ 是后验均值，在这个语境下，$\mathrm{E}[w]$ 也叫作贝叶斯点（这种插值估计低估了不确定性）



##### 8.4.4.1 蒙特卡罗方法近似

更好的近似方法就是蒙特卡罗方法，其定义如下：
$$
p(y=1|x,D)\approx \frac{1}{S}\sum^S_{s=1} sigm((w^s)^Tx)
$$
其中，$w^s\sim p(w\|D)$ 是在后验中的取样

* 如果使用蒙特卡罗方法估计后验，就可以复用这些样本来进行预测.
* 如果对后验使用高斯估计，就要用标准方法从高斯分布中取得独立样本



### 8.5 在线学习 (Online learning) 和随机优化 (stochastic optimization)

传统机器学习都是线下的，也就意味着是有一个批量的数据，然后优化一个下面形式的等式：
$$
f(\theta)=\frac{1}{N}\sum^N_{i=1}f(\theta,z_i)
$$
其中 $z_i=(x_i,y_i)$ 是监督学习情况，或者只有 $x_i$ 对应着无监督学习的情况，而$f(\theta,z_i)$这个函数是某种损失函数

例如可以使用下面的损失函数：
$$
f(\theta,z_i)=L(y_i,h(x_i,\theta))
$$
其中的$h(x_i,\theta)$是预测函数，而$L(y,\hat y)$是某种其他的损失函数

> 在频率论统计学方法中,平均损失函数也叫作风险，所以对应地就将这个方法整体叫做经验风险最小化

可是如果有一系列的流数据不停出现，就需要进行在线学习，也就是要随着每次有新数据来到而更新估计；另外有时候虽然数据是成批的一整个数据，也可能会因为太大没办法全部放进内存等原因也需要使用在线学习、



#### 8.5.1 在线学习和遗憾最小化（Online learning and regret minimization）

假如在每一步中，客观世界都提供了一个样本$z_k$，而学习者必须使用一个参数估计$\theta_k$对此进行响应。在线学习关注的目标是遗憾值 (regret)，定义为相对于使用单个固定参数值时候能得到的最好结果所得到的平均损失：

$$
regret_k\overset{\triangle}{=} \frac{1}{k} \sum^k_{t=1} f(\theta_t,z_t)-\min_{\theta^*\in \Theta}\frac{1}{k} \sum^k_{t=1}f(\theta_*,z_t)
$$

> 比如需要调查股票市场，设$\theta_j$是我们在股票 $j$ 上面投资的规模，而 $z_j$ 表示这个股票带来的回报.，则损失函数为 $f(\theta,z)=-\theta^Tz$，遗憾值就是我们通过每次交易而得到的效果



在线学习的简单算法是在线梯度下降法，步骤如下：

> 在每次第 $k$ 步，使用下列表达式更新参数
> $$
> \theta_{k+1}= \text{proj}_{\Theta}(\theta_k-\eta_kg_k) \ \text{(8.78)}
> $$
> 其中，$\text{proj}_v(v)=\arg\min_{w\in V}\|\|w-v\|\|_2$ 是向量 $v$ 在空间 $V$ 上的投影，$g_k=\nabla f(\theta_k ,z_k)$是梯度项，$\eta_k$ 是步长
>
> NOTE：只有当参数必须要约束在某个$R^D$的子集内的时候才需要使用投影这个步骤



#### 8.5.2 随机优化和风险最小化

接下来将要尝试的不是让过去步骤的遗憾最小化，而是希望未来损失最小化，即要最小化：
$$
f(\theta )=\mathbb{E}[f(\theta,z)] \ \text{(8.79)}
$$
其中这个期望是在未来数据上获取的，优化这种某些变量是随机变量的函数的过程就叫做随机优化

假设要从一个分布中接收到无限的样本流，优化等式 8.79 里面的期望值的一个方法就是在每一步应用等式 8.78 进行更新，这就叫做随机梯度下降（SGD）

通常我们都想要一个简单的参数估计，可以用下面的方法进行平均：
$$
\bar\theta_k=\frac{1}{k}\sum^k_{t=1}\theta_t
$$
这被称为 Polyak-Ruppert 平均，可以递归使用：
$$
\bar\theta_k=\bar\theta_{k-1}-\frac{1}{k}( \theta_{k-1}-\theta_k)
$$

##### 8.5.2.1 设定步长规模

接下来要讨论的是要保证随机梯度下降收敛所需要的学习速率的充分条件（Robbins-Monro 条件）：
$$
\sum^\infty_{k=1}\eta_k=\infty,\ \sum^\infty_{k=1}\eta^2_k <\infty
$$
$\eta_k$ 在时间上的取值集合也叫作学习速率列表，可以使用很多公式，如 $\eta_k = 1/k$，或者
$$
\eta_k = (\tau_0 + k) ^{-\kappa}
$$
其中，$\tau_0 \ge 0$ 减慢了算法的早期迭代，而 $\kappa\in (0.5,1]$ 控制了旧值被遗忘的速率

随机优化的一个主要缺陷就是需要去调整这些参数，一个简单的启发式办法如下所示:

> 存储数据的一个初始子集，然后对这个子集应用一系列不同的 $\eta$ 值，然后选择能使得目标对象降低最快的，再将其应用于其他的全部数据上
>
> NOTE：这可能会导致不收敛，但当在留出集上的性能改进趋于平稳时，算法就可以终止了（这被称为提前停止）



##### 8.5.2.2 每个参数的步长（Per-parameter step sizes）

随机梯度下降法的一个缺点就是对于不同的参数都用同样的步长规模，下面将介绍自适应梯度下降法（adagrad），这个方法的思路类似于使用对角海森矩阵近似，具体来说就是如果 $\theta_i(k)$ 是第 $k$ 次的参数 $i$，而 $g_i(k)$ 是对应的梯度，就可以使用下面的方式进行更新：
$$
\theta_i(k+1)=\theta_i(k)-\eta\frac{g_i(k)}{\tau_o+\sqrt{s_i(k)}}
$$
其中，对角线步长向量是梯度向量的平方，是所有时间步长的总和，可以使用下面的形式来递归更新：
$$
s_i(k)=s_i(k-1)+g_i(k)^2
$$
这个结果就是一个分参数步长规模，可以适应损失函数的曲率



##### 8.5.2.3 随机梯度下降和批量学习的对比

如果没有一个无限的数据流，可以去模拟一个，只要随机从训练集中取样数据点即可

随机梯度下降的步骤：

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051739014.png)

理论上应该是有放回取样，不过实际上通常都是随机排列数据，进行不替换抽样，然后重复进行的效果更好；然后后面都重复这样操作，每次对全部数据集进行一次抽样就叫做一个 epoch

在线下学习的情况下,更好的方法是以 B 份数据来进行小批量梯度计算

* 如果B=1，这就是标准的随机梯度下降法
* 如果B=N，这就成了标准的梯度下降法（最陡下降）
* 一般都设置$B\sim 100$



随机梯度下降法的优点：

* 虽然随机梯度下降法是很简单的一阶方法，但用于一些问题的时候效果出奇地好，尤其是数据规模很大的情况（直观理解起来，原因可能是只要看过几个样本之后就能对梯度进行很好的估计，而使用很大规模的数据来仔细计算精准的梯度就很可能是浪费时间）
* 除了能加速之外，随机梯度下降法还不太容易在浅局部最小值部位卡住，因为通常增加了一定规模的噪音



#### 8.5.3 最小均方算法

举个随机梯度下降法的例子：

> 假设要考虑去计算在线学习中线性回归的最大似然估计，在第 $k$ 次迭代的在线梯度为：
> $$
> g_k =x_i(\theta^T_kx_i-y_i)
> $$
> 其中的$i=i(k)$是第k次迭代时候使用的训练样本，如果数据集是流式的，就用$i(k)=k$

上面的等式很好理解：特征向量 $x_k$ 乘以预测值 $\hat y_k =\theta^T_k x_k$ 和真实的响应变量 $y_k$ 的差距作为权重，因此梯度函数就像是一个误差信号

在计算了梯度之后，可以沿着梯度进行下一步长的前进：
$$
\theta_{k+1}=\theta_k-\eta_k(\hat y_k-y_k)x_k
$$
（非约束优化问题，不需要投影步骤）这个就叫做最小均方算法，也被称作 $\delta$ 规则或者 Widrow-Hoff 规则

**NOTE**：最小均方算法可能需要很多次遍历整个数据才能找到最优解；对比之下，基于卡尔曼过滤器的递归最小均方算法使用二阶信息只需要单次遍历数据就能找到最优解



#### 8.5.4 感知器算法

接下来考虑如何对在线情况下的二值化逻辑回归模型进行拟合，在线情况下的权重更新具有简单的形式：
$$
\theta_k=\theta_{k-1}-\eta_kg_i =\theta_{k-1} -\eta_k(\mu_i-y_i)x_i
$$
其中 $\mu_i=p(y_i=1\|x_i,\theta_k)=\mathrm{E}[y_i\|x_i,\theta_k]$.

然后对这个算法进行近似，设:
$$
\hat y_i =\arg\max_{y\in\{0,1\}} p(y|x_i,\theta)
$$
代表了最大概率类标签

将梯度表达式中的 $\mu_i =p(y=1\|x_i,\theta)=sigm(\theta^Tx_i)$ 替代为$\hat y_i$，这样就得到了近似的梯度：
$$
g_i\approx (\hat y_i -y_i)x_i
$$
如果我们假设$y\in \{-1,+1\}$，而不是$y\in \{0,1\}$,那么在代数计算上就能更简单了.这时候我们的预测就成了:
$$
\hat y_i =sign(\theta^Tx_i)
$$
然后如果$\hat y_i y_i =-1$，就分类错误了，而如果 $\hat y_i y_i =+1$ 则表示猜对了分类标签.

在每一步都要通过加上梯度来更新权重向量，关键的观测项目在于，如果预测正确，那么$\hat y_i=y_i$，所以(近似)梯度就是零，也就不用去更改权重向量；可是如果$x_i$是误分类的，就要按照下面的步骤更新权重向量了：

* 如果 $\hat y_i =1$ 而 $y_i = -1$，那么负梯度就是$-(\hat y_i-y_i)x_i=-2x_i$；
* 如果反过来 $\hat y_i =-1$ 而 $y_i = 1$，那么负梯度就是$-(\hat y_i-y_i)x_i= 2x_i$.

上面这个因数2可以吸收进学习速率 $\eta$ 里面，则在误分类的情况下更新的形式为:
$$
\theta_k =\theta_{k-1}+\eta_k y_i x_i
$$
由于只有权重的符号是重要的，而大小并不重要，所以就可以设$\eta_k=1$

上面这个算法叫做感知器算法，在给的数据是线性可分的情况下就会收敛，即存在参数$\theta$使得在训练集上的预测$sign(\theta^Tx)$会达到零误差；不过如果数据不是线性可分的，这个算法就不收敛了，甚至可能虽然收敛也要花费很长时间才行

**感知器算法**

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051739654.png)



#### 8.5.5 贝叶斯视角

在线学习的另外一个方法就是从贝叶斯视角实现的，这个概念特别简单，就是递归应用贝叶斯规则：
$$
p(\theta|D_{1:k})\propto p(D_k|\theta)p(\theta|d_{1:k-1})
$$

* 这有个很明显的优势，因为返回的是一个后验，而不是一个点估计，这也允许超参数的在线自适应（这是非常重要的，因为在线学习没办法使用交叉验证）
* 最后的一个不那么明显的优点就是速度可以比随机梯度下降法更快。具体原因在于，通过对每个参数加上其均值的后验方差建模，可以有效对每个参数赋予不同的学习速率，这是对空间曲率建模的一种简单方法。这些房差通过概率论的常见规则就可以实现自适应。对比之下，使用二阶优化方法来解决在线学习的问题可能就要麻烦多了

