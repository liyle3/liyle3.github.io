---
redirect_from: /_posts/2024-06-25-Multi_Fidelity.md
title: Multi-Fidelity
tags:
  - 论文阅读
---



**Deep Multi-Fidelity Active Learning of High-Dimensional Outputs **（AISTATS）

<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051931309.png" style="zoom:67%;" />

* setting

  * high fidelity $\longrightarrow$ high-resolution (high output dimension)

  * fixed cost $\lambda_m$ for fidelity $m$

    

* surrogate model

  * $y_m(x) = f(x, m)$
  * stochastic structural variational learning algorith：输出维度过大，MC-Dropout不可行

    

* acquisition function
  $$
  a(x, m) = \frac{1}{\lambda_m} \mathbb{I}(y_m(x), y_M(x)| \mathcal{D}) \\
  = \frac{1}{\lambda_m} (\mathbb{H}(y_m | \mathcal{D}) + \mathbb{H}(y_M | \mathcal{D}) - \mathbb{H}(y_m, y_M|\mathcal{D}))
  $$

  * 基于保真度 $m$ 和 $M$ 上模型输出的互信息熵

  * $y_m$为对于保真度$m$的模型预测结果，$M$为最高保真度

  * 传统AL的Entropy策略的扩展：$m=M$时，$\alpha(x, M) = \frac{1}{\lambda_M} \mathbb{H}(y_M(x) | \mathcal{D})$

    

<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051931574.png" style="zoom:67%;" />



**fidelity & instance - wise **

* data point：$\mathcal{D} = \{(x_t, m, \lambda_{t, m})\}$

  

* surrogate model $\mathcal{SM}$

  * $y_m(x_t) = \mathcal{SM}(x_t, m)$

  * model the posterior $p(f_m(x_t)| x_t, m, \mathcal{D})$，($f_m$：oracle with m-fidelity)

    

* acquisition function $\alpha(x_t, m)$

  * data selection：$(x_t, m) = \underset{x_t \in \mathcal{X}, \ 1 \le m\le M}{\operatorname{argmax}} \alpha(x_t, m)$
  * cost function $\mathcal{CF}$：$\lambda_{t, m} = \mathcal{CF} (x_t, m)$
  * update：$\mathcal{D} = \mathcal{D} \ \cup \{(x_t, m, \lambda_{t, m})\}$



**MF-CAL**

* 对于未标记的样本，使用KNN获取其周围K个有标记的样本，并计算不确定分数 $S$
  $$
  S = \alpha(x, m) = \frac{1}{|N(x)| \ \lambda_m}\sum _{x_i \in N(x)} \lambda_{m, i} (\hat f(x_i, M) - y_i)^2
  $$

* 采样
  $$
  (x, m) = \underset{x_t \in \mathcal{X}, \ 1 \le m\le M}{\operatorname{argmax}} \alpha(x, m)
  $$

**MF-suggestive annotation**

* 先选出一批具有高不确定性的样本（ensemble）
* 再对利用cost对不确定分数进行加权，选出分数高的部分样本



