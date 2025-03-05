---
redirect_from: /_posts/2023-09-25-Prior and Posterior Networks A Survey on Evidential Deep Learning Methods For Uncertainty Estimation .md
title: Prior and Posterior Networks
tags:
  - 论文阅读
---





# Notes for Prior and Posterior Networks: A Survey on Evidential Deep Learning Methods For Uncertainty Estimation 



## 1. Motivation

* BNN uncertainty estimation：Monte Carlo dropout, Markov Chain sampling, ensembling
* EDL: 
  * single model and forward pass
  * know what is unknown: distributional uncertainty, OOD fall back onto a prior belief



## 2. Definition

* **Bayesian model averaging (BMA)**

  <img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051929412.png" style="zoom:67%;" />

* **Evidential Deep Learning (EDL)**

  <img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051929520.png" style="zoom:67%;" />

  * replace $p(\theta | D)$ by a point estimate $\hat \theta$ using the Dirac delta function
  * The advantage of this approach is further that it allows us to distinguish uncertainty about a data point because it is ambiguous from points coming from an entirely different data distribution.





## 3. Uncertainty Estimation

* **Data (aleatoric) uncertainty**

  <img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051929585.png" style="zoom:67%;" />

* **Model (Epistemic) Uncertainty**

* **Distributional uncertainty**: uncertainty caused by the mismatch of training and test data distributions

  <img src="https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051929225.png" style="zoom:67%;" />

   **Distinguishing epistemic from distributional uncertainty also allows us to differentiate uncertainty due to underspecification from uncertainty due to a lack of evidence**



## 4. Existing Approaches for Dirichlet Networks

### 4.1 Prior Networks

* **Challenge**

  * ensure high classification performance

  *  the intended behavior under OOD inputs

    

* **OOD-free approaches**

* **Knowledge distillation**

* **OOD-dependent approaches**

* **Sequential models**



### 4.2 Posterior Networks

* **Generating OOD samples using generative models**
* **Posterior networks via Normalizing Flows**
* **Posterior networks via variational inference**





## 5. Discussion

### 5.1 BMA

### 5.2 Challenges

* many of the loss functions used so far are not appropriate and violate basic asymptotic assumptions about epistemic uncertainty: With increasing
  amount of data, epistemic uncertainty should vanish, but this is not guaranteed using the commonly used loss functions. 
*  some approaches require out-of-distribution data points during training
* cannot explicitly estimate epistemic uncertainty

