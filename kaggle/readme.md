### 本程序食用指南

#### 导入包
请首先在空的`.py`文件中运行以下函数：
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# cluster_1 = cluster_1.drop('Churn', axis=1)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as skLDA
```
其中`seaborn`包需要特别另外安装，它用于美化输出的数据分析图表。

#### 运行之前
1.请确保`main.py`函数与数据文件`WA_Fn-UseC_-Telco-Customer-Churn.csv`在同一目录下，否则可能需要修改第`273-274`行代码的路径字符串。数据集来源：[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

2. 在`155-162`行代码附有`sklearn`自带的`LDA`模型函数与预测评估。若作为对比，可以将这部分代码取消注释以运行，查看其模型准确率。同理对`LogisticRegression`有第`179-182`行代码的预设函数，可以进行类似的对比观察。

3. 若需查看作图函数，请解除第`274`行代码的注释。

4. 在`LogisticRegression`中，由于可能存在的`np.exp()`数值爆炸的情况，导致最终算出的查准率、查全率等为`nan`。这并非程序运行错误，而是我们的算法设计在短时间内没能解决这一问题。关于这个问题，我们在论文中有解释。

5. 所有**算法本身相关**的函数均为从零开始实现。若存在高度封装的预设模型函数，均在注释中。

#### 运行
直接运行本`main.py`文件即可。