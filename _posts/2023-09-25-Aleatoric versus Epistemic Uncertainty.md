---
redirect_from: /_posts/2023-09-25-Aleatoric versus Epistemic Uncertainty.md
title: Aleatoric versus Epistemic Uncertainty
tags:
  - 学习笔记
---



# Aleatoric versus Epistemic Uncertainty



**Epistemic Uncertainty（认知不确定性）**：模型中存在的不确定性，模型对某些数据（或区域）缺乏认知（lack of knowledge），or 输入数据是否存在于已经见过的数据的分布之中

**Aleatorci Uncertainty（偶然不确定性）**：由于观测数据中的固有噪声导致的



**区分标准：**是否能通过增加数据量来降低不确定性



几种情况

* **Low E + Low A**

  模型学习到该区域的数据分布，数据点距离决策边界较远，模型预测置信度高  ----》对应 easy example

  

* **Low E + High A**

  该区域样本数据量充足（模型学习到数据分布），但数据点距离决策边界很近，模型预测置信度低，误分类的概率高 ---》对应 hardness or uncertainty



* **High E + Low A**

  该区域样本数据较少，模型对该区域分布缺乏认知，模型预测不可靠

  如：OOD example 或有标签样本少  ---》 对应diversity



* **High E + High A**

  与上一种情况相同，Aleatorci Uncertainty 可被忽略







![](https://raw.githubusercontent.com/liyle3/picgo-resources/main/202503051921147.png)
