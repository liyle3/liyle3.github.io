---
redirect_from: /_posts/2024-06-25-DFT-related-work.md
title: DFT related work
tags:
  - 论文阅读
---



# A transferable recommender approach for selecting the best density functional approximations in chemical discovery



## 1. Motivation

* 没有通用精度的DFA，导致DFT生成的数据质量具有不确定性
* DFA recommender 选择与 gold standard ( DLPNO-CCSD(T) ) 之间具有最小误差的DFA



## 2. Method

**总体架构**

* 使用$\Delta$-learning模型以回归任务的方式预测每种DFA（共有48种）的计算值与 gold standard 之间的绝对差值 $|\Delta \Delta E_{H-L}[f]|$
* 根据$|\Delta \Delta E_{H-L}[f]|$来推荐DFA



**Pipeline**

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051933342.png)





* **电子密度（electron density）**
  $$
  \rho _{{HS}^{\alpha}}(r) - \rho _{{LS}^{\alpha}}(r) = \sum _A \sum _Q C_Q^{(A, \alpha)} \phi_Q(r - r_A) \\
  \rho _{{HS}^{\beta}}(r) - \rho _{{LS}^{\beta}}(r) = \sum _A \sum _Q C_Q^{(A, \beta)} \phi_Q(r - r_A) \\
  $$

  * $\alpha$：majority spin
  * $\beta$：minority spin
  * $\phi_Q(r - r_A)$：原子A 第Q个DF基函数
  * $C_Q^{(A, \alpha)}$：基函数的系数

  

* **模型输入$p_L^A$**

  * 分别计算$\alpha$和$\beta$的功率谱
    $$
    p_L^{A, \alpha} = \sum _{Q \in L} ||C^{(A, \alpha)}_Q||^2 \\
    p_L^{A, \beta} = \sum _{Q \in L} ||C^{(A, \beta)}_Q||^2 \\
    $$

  * 将$p_L^{A, \alpha}$和$p_L^{A, \beta}$拼接得到模型输入特征$p_L ^A$

  

* $\Delta-\text{learning}$ **模型**：Behler–Parrinello-type neural networks，预测 $|\Delta \Delta E_{H-L}[f]|$
  $$
  X_A ^l = \sigma (W_{A \in g} ^l X_A ^{l-1})
  $$

  * $X_A^l$：第 l 层原子 A 的表征
  * $g$：元素族，同一族的元素将使用相同的模块与参数

  * 最后一层中每种元素的表征由求和得到
    $$
    X_e^n = \sum _{A \in e} X_A ^n
    $$

  * 不同元素的表征拼接并输入全连接层得到最后的输出

  * 每个DFA对应一个$\Delta$-learning 模型



* **Recommender**
  $$
  f_{rec} = \arg \min _{f \in F} ||\Delta \Delta E_{H-L}[f]||
  $$
  





### 3. Experiment

**The approach is demonstrated on the evaluation of vertical spin splitting energies of transition metal complexes**

* **Dataset**：VSS-452
* **split**
  * train：66%，300 points
  * valid：60 points in train
  * test：34%，152 points



![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051933572.png)



**迁移性实验**

* **OOD dataset**：CSD-76

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051934273.png)





# Supervised learning of a chemistry functional with damped dispersion



## 1. Motivation

* 没有可以准确预测所有化学性质的泛函
* 双杂化泛函的发展比较接近这个目标
* 本文优化了一种单杂化泛函 CF22D
  * 将取决于密度和占据轨道的全局混合元不可分离梯度近似与取决于几何形状的阻尼色散项相结合
  * 比大多数现有的非双杂化泛函具有更高的全面准确性





## 2. Method

* **functional form**
  $$
  E^{CF22D} = E_{DF} + E_{disp}
  $$

  * $E_{DF}$：交换关联项，具有 MN15 泛函的函数形式
  * $E_{disp}$：分子力学项，也称为阻尼色散项

* **training process**

  1. 选取初始训练集，初始化参数
  2. 训练集的电子密度由 MN15 泛函计算得到，作为初始密度
  3. CF22D中的每个描述子都可以基于上一步的电子密度计算得到
  4. 使用广义约化梯度非线性算法最小化损失函数
  5. 使用上一步得到的试验泛函来计算验证集和测试集的能量和MUE
  6. 监督学习步骤：如果在验证集的某个数据集中，试验泛函的MUE比该数据集上top-5泛函的平均MUE高出30%，则将该数据集加入训练集
  7. 如果由数据集加入训练集，则重新计算训练集的能量，并回到步骤3；否则将训练集的MUE与前几次的迭代相比较，若收敛则结束训练过程



![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051934090.png)





## 3. Result

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051934683.png)

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051934624.png)





# Multi-Fidelity Active Learning with GFlowNets

## 1. Motivation

* 高保真度的oracles开销太昂贵
* 利用不同保真度的oracles进行active learning，兼顾成本开销和样本多样性 (GFlowNet)
* 选出具有高分数的目标对象

Multi-Fidelity Active Learning with GFlowNets

## 2. Method

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051934840.png)



* **代理函数 h**：建模后验分布 $p(f_m(x)|x, m, D)$，Deep Kernel Learning

  

* **GFlowNet**

  * 使用获取函数$\alpha(x, m)$作为奖励函数，输出为与奖励函数成比例的分布 $\pi_{\theta}(x)$
    $$
    R(\alpha(x, m), \beta) = \frac{\alpha(x, m) \times \rho ^{j-1}}{\beta}
    $$
    

* **获取函数$\alpha(x, m)$​**：Max-value Entropy Research（最后使用 GIBBON 近似）

$$
\alpha(x, m) = \frac{1}{\lambda_m} I(f^{\star} _M;f_m|D_j)
$$



**Pipeline**

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051934786.png)



## 3. experiment

* **Mean top-K score：**mean score, per the highest fidelity oracle fM , of the top-K samples
* **Top-K diversity：**mean pairwise distance within the top-K samples

**multi-fidelity approaches are not aimed at achieving better mean top-K scores than a single-fidelity active learning counterpart, but rather the same mean top-K scores but with a smaller budget**



### 3.1 DNA APTAMERS

* **Object**：最大化 DNA 序列二级结构的（负）自由能
* **Diversity** is computed as one minus the mean pairwise sequence identity among a set of DNA sequences
* **Dataset**：construct a test set by sampling sequences from a uniform distribution of the free energy
* **Oracle**
  * Highest fidelity：NUPACK（software），cost = 20
  * lower fidelity：Transformer model on 1 million randomly sampled sequences annotated with $f_M$，cost = 0.2

* **Setting**
  * consider fixed-length sequences of 30 bases
  * batch size = 512





<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051934961.png" style="zoom:80%;" />





### 3.2 ANTIMICROBIAL PEPTIDES

>  short protein sequences which possess antimicrobial properties

* **Object**: identify sequences with a high antimicrobial activity

* **Diversity**：与DNA相同

* **Dataset**: DBAASP，划分为三个部分，$D_1$训练oracle，$D_2$作为初始数据集，$D_3$作为测试集

* **Oracle**：三个不同的神经网络模型

  * $f_M$,  $\lambda_M = 50$
  * $f_1$,   $\lambda_1 = 0.5$
  * $f_2$,   $\lambda_2=0.5$

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051934209.png)

* **Setting**
  * consider variable-length protein sequences with up to 50 residues
  * batch size = 32





<img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051934081.png" style="zoom:80%;" />





### 3.3  SMALL MOLECULES

* **Object**: consider two proof-of-concept tasks in molecular electronic potentials: maximisation of the (negative) adiabatic ionisation potential (IP) and of the adiabatic electron affinity (EA)

* **Setting**
  * designed the GFlowNet state space by considering variable length sequences of SELFIES tokens to represent molecules
  * vocabulary size = 26
  * maximum length = 64
  * batch size = 128
* **Dataset**：1400 molecules
* **Oracle**: 三种不同级别
  * $f_M$：使用RDKit得到的四个构象异构体，并通过MMFF94优化后取能量最低的构象异构体，并通过GFN2-xTB进一步优化；$\lambda_M = 7$
  * $f_1$：使用RDKit得到的一个构象异构体，其几何形状通过力场 MMFF94 进行优化；$\lambda_1 = 1$
  * $f_2$：使用RDKit得到的两个构象异构体，并通过MMFF94优化后取能量最低的构象异构体，并通过GFN2-xTB进一步优化以获得基态几何结构；$\lambda_2 = 7$



<img src="D:\DESKTOP\ML\projects\CAMFALAD\notes\figures\resultc.png" style="zoom:80%;" />





## OC20

* 1,281,040 DFT松弛 （264,890,000 单点估计）
* tasks
  * Structure to Energy and Forces (S2EF)
  * Initial Structure to Relaxed Structure (IS2RS)
  * Initial Structure to Relaxed Energy (IS2RE)
* baseline model：CGCNN, SchNet, DimeNet++





















































