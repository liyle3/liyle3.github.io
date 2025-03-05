---
redirect_from: /_posts/2024-08-02-Survey Generative Models for Graph.md
title: Survey on Generative Models for Graph
tags:
  - 论文阅读
---

# Survey: Generative Models for Graph

by liyle3



## Sampling strategies

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948623.png)





## Generation Strategies

* **One-hot generation**

  > One-shot generation usually generates a new graph represented in an adjacency matrix with optional node and edge features in one single step

* **Sequential generation**

  > In contrast to one-shot generation, sequential generation generates a graph consecutively in a few steps





## Summary

![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948332.png)



## VAE

#### GraphVAE ([ICANN 2018](https://arxiv.org/pdf/1802.03480))

* **latent space**: $N(0, I)$
* **dataset**
  * QM9
  * ZINC
  * 训练时会将节点特征预处理为one-hot编码
* **[code](https://github.com/guydurant/GraphVAE)**

* **pipeline**

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948455.png)



#### PGD-VAE ([NIPS 2022](https://arxiv.org/pdf/2201.11932))

* for Periodic Graphs (crystal nets and polygon mesh)

* **Pipeline**

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948213.png)



#### D-MolVAE ([Bioinformatics 2022](https://cs.emory.edu/~lzhao41/materials/papers/Small_Molecule_Generation_via_Disentangled_Representation_Learning_Bioinformatics__Copy_%20(1).pdf))

* no code
* for molecules



#### MDVAE ([SDM 2022](https://arxiv.org/pdf/2203.00412))

* *node feature is an one-hot vector encoding the type of atom*

* [code](https://github.com/yuanqidu/MDVAE)

* **latent space**: Isotropic Gaussian

  * Each $Z_i$ is independent of the others
  * Each dimension of $Z$ represents one real property

* **dataset**

  * QM9
  * ZINC

* **Pipeline**

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948263.png)



#### NED-VAE ([KDD 2020](https://arxiv.org/pdf/2006.05385))

* [code](https://github.com/xguo7/NED-VAE) 

* **dataset**

  * **Erdos-Renyi Graphs**
  * **Watts Strogatz Graphs**
  * **Protein Structure Dataset**

* **latent space**：Gaussian distribution

  * $Z_f: \mu _f + \sigma _f \ \odot \ \epsilon, \text{where } \epsilon \text{ is a standard normal distribution}$
  * ....

* **pipeline**

  <img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948172.png" style="zoom:100%;" />



#### SND-VAE ([KDD 2021](https://cs.emory.edu/~lzhao41/materials/papers/KDD21__Spatial_Graphs_Disentanglement_preprinted.pdf))

* [code](https://github.com/xguo7/SND-VAE) 

* for spatial network (each node have a 2D/3D geometric coordinate)

* **Pipeline**

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948827.png)



#### **D2G2** ([SDM21](https://arxiv.org/pdf/2010.07276))

* for dynamic graphs



#### STGD-VAE ([arxiv 2022](https://arxiv.org/pdf/2203.00411))

* Disentangled **Spatiotemporal** Graph Generative Models



#### HierVAE ([ICML 2020](https://arxiv.org/pdf/2002.03230))

* Hierarchical Generation of Molecular Graphs using Structural Motifs

* **Pipeline**

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948365.png)





## Diffusion

#### EDP-GNN ([AISTATS 2020](https://arxiv.org/pdf/2003.00638))

* *reconstruct adjacency matrix only*

* **[code](https://github.com/ermongroup/GraphScoreMatching)**

* **dataset**

  ![](./figures/EDPGNN-dataset.png)

* **Pipeline**

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948283.png)



#### MOOD ([arxiv 2022](https://arxiv.org/pdf/2206.07632))

* Exploring Chemical Space with Score-based **Out-of-distribution** Generation



#### GDSS ([ICML 2022](https://arxiv.org/pdf/2202.02514))

* **[code](https://github.com/harryjo97/GDSS)**
* **dataset**
  * Ego-small：4 ≤ |V | ≤ 18
  * Community-small：12 ≤ |V | ≤ 20
  * Enzymes：10 ≤ |V | ≤ 125
  * Grid：100 ≤ |V | ≤ 400
  * QM9
  * ZINC250k
* **latent space**：Gaussian distributions where the mean and variance are tractably determined by the coefficients of the forward diffusion process
* **random sample**



#### DiGress ([ICLR 2023](https://arxiv.org/pdf/2209.14734))

* **[code](https://github.com/cvignac/DiGress)**
* **latent space**：$q_X \times q_E$

* **dataset**

  * SBM
  * QM9
  * MOSES
  * GuacaMol

* **Pipeline**

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948769.png)



## Normalized Flow

#### GraphNVP ([arxiv 2019](https://arxiv.org/pdf/1905.11600)) GRF ([arxiv 2019](https://arxiv.org/pdf/2203.06714))

* Problem formulation: $G = (A, X)$

  * $A$：adjacency matrix
  * $X$：feature matrix, $X \in \{0, 1\}^{N \times M}$, $M$ is the number of types of nodes

   





## VQVAE variant

#### DGAE ([arxiv 2024](https://arxiv.org/pdf/2306.07735))

* **[code](https://github.com/yoboget/dgae)**

* **dataset**

  * Ego-Small
  * Community-Small
  * Enzymes
  * QM9
  * ZINC

* **Pipeline**

  ![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051948711.png)

