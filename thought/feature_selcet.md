# 特征选择

特征选择是特征工程里的一个重要问题，其目标是**寻找最优特征子集**。特征选择能剔除不相关(irrelevant)或冗余(redundant )的特征，从而达到减少特征个数，**提高模型精确度，减少运行时间的目的**。另一方面，选取出真正相关的特征简化模型，协助理解数据产生的过程。并且常能听到“数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已”，由此可见其重要性。但是它几乎很少出现于机器学习书本里面的某一章。然而在机器学习方面的成功很大程度上在于如何使用特征工程。



## 三大类方法

根据特征选择的形式，可分为三大类：

- Filter(过滤法)：按照**发散性**或**相关性**对各个特征进行评分，设定阈值或者待选择特征的个数进行筛选
- Wrapper(包装法)：根据目标函数（往往是预测效果评分），每次选择若干特征，或者排除若干特征
- Embedded(嵌入法)：先使用某些机器学习的模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征（类似于Filter，只不过系数是通过训练得来的）



## 过滤法

基本想法是：分别对每个特征 xi ，计算 xi 相对于类别标签 y 的信息量 S(i) ，得到 n 个结果。然后将 n 个 S(i)  按照从大到小排序，输出前 k 个特征。显然，这样复杂度大大降低。那么关键的问题就是使用什么样的方法来度量 S(i) ，我们的目标是选取与 y  关联最密切的一些 特征xi 。

#### Pearson相关系数

皮尔森相关系数是一种最简单的，能帮助理解特征和响应变量之间关系的方法，衡量的是变量之间的线性相关性，结果的取值区间为[-1,1]， -1 表示完全的负相关(这个变量下降，那个就会上升)， +1 表示完全的正相关， 0 表示没有线性相关性。Pearson Correlation速度快、易于计算，经常在拿到数据(经过清洗和特征提取之后的)之后第一时间就执行。

使用 **Pearson** 相关系数之前需要检查数据是否满足前置条件：

1. 两个变量间有线性关系；
2. 变量是连续变量；
3. 变量均符合正态分布，且二元分布也符合正态分布；
4. 两变量独立；
5. 两变量的方差不为 0；

**Pearson** 相关系数对异常值比较敏感。

```python
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest
# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
# 参数k为选择的特征个数
# Y保证shape是（N,）, f_regression好像原理就是这个
# careful 选出来的特征并不是按照分数高低有序排列的，只是保证了分数前k大
SelectKBest(lambda X, Y: list(array(map(lambda x:pearsonr(x, Y), X.T)).T), k=2).fit_transform(iris.data, iris.target)
```

#### 卡方验证

经典的卡方检验是检验**类别型变量**对**类别型变量**的相关性。假设自变量有N种取值，因变量有M种取值，考虑自变量等于i且因变量等于j的样本频数的观察值与期望的差距

```python
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target  #iris数据集

#选择K个最好的特征，返回选择特征后的数据
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
```

#### 互信息和最大信息系数

想把互信息直接用于特征选择其实不是太方便：

1. 它不属于度量方式，也没有办法归一化，在不同数据及上的结果无法做比较
2. 对于连续变量的计算不是很方便（ X 和 Y 都是集合, xi, y 都是离散的取值），通常变量需要先离散化，而互信息的结果对离散化的方式很敏感

**最大信息系数**克服了这两个问题。它首先寻找一种最优的离散化方式，然后把互信息取值转换成一种度量方式，取值区间在 [0,1] 。minepy提供了MIC功能。

```python
from minepy import MINE
from sklearn.feature_selection import SelectKBest
#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
# 选择K个最好的特征，返回特征选择后的数据
SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)
```

#### 距离相关系数

距离相关系数是为了克服Pearson相关系数的弱点而生的。在 x 和 x^2 这个例子中，即便Pearson相关系数是 0 ，我们也不能断定这两个变量是独立的（有可能是非线性相关）；但如果距离相关系数是 0 ，那么我们就可以说这两个变量是独立的。另外这是[Python gist](https://link.zhihu.com/?target=https%3A//gist.github.com/josef-pkt/2938402)的实现。

#### 方差选择法

过滤特征选择法还有一种方法不需要度量特征 x_i 和类别标签 y 的信息量。这种方法先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。

```python
from sklearn.feature_selection import VarianceThreshold
# 方差选择法，返回值为特征选择后的数据
# 参数threshold为方差的阈值
sel = VarianceThreshold(threshold=(t))
print(sel.fit_transform(X))
```

其他还有很多很多过滤方法，等以后有机会学习特征工程时候再补充

![feature_selcet_filter_method](https://raw.githubusercontent.com/crazy-JiangDongHua/dl/master/thought/feature_selcet_filter_method.png)



## 包装法

基本思想：基于hold-out方法，对于每一个待选的特征子集，都在训练集上训练一遍模型，然后在测试集上根据误差大小选择出特征子集。需要先选定特定算法，通常选用普遍效果较好的算法， 例如Random Forest， SVM， kNN等等。

#### 前向搜索

前向搜索说白了就是，每次增量地从剩余未选中的特征选出一个加入特征集中，待达到阈值或者 n 时，从所有的 F 中选出错误率最小的。过程如下：

1. 初始化特征集 F 为空。
2. 扫描 i 从 1 到 n，如果第 i 个特征不在 F 中，那么特征 i 和F 放在一起作为 Fi 。在只使用 Fi 中特征的情况下，利用交叉验证来得到 Fi 的错误率。
3. 从上步中得到的 n 个 Fi 中选出错误率最小的 Fi ,更新 F 为 Fi 。
4. 如果 F 中的特征数达到了 n 或者预定的阈值（如果有的话），那么输出整个搜索过程中最好的 ；若没达到，则转到 2，继续扫描。

#### 后向搜索

既然有增量加，那么也会有增量减，后者称为后向搜索。先将 F 设置为 {1,2,...,n} ，然后每次删除一个特征，并评价，直到达到阈值或者为空，然后选择最佳的 F 。

这两种算法都可以工作，但是计算复杂度比较大。时间复杂度为：![[公式]](https://www.zhihu.com/equation?tex=O%28n%2B%28n-1%29%2B%28n-2%29%2B...%2B1%29%3DO%28n%5E2%29+) 

#### 递归特征消除法

递归消除特征法使用一个**基模型**来进行多轮训练，每轮训练后通过学习器返回的 coef_ 或者feature_importances_ 消除若干权重较低的特征，再基于新的特征集进行下一轮训练。

使用feature_selection库的RFE类来选择特征的代码如下：

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#递归特征消除法，返回特征选择后的数据
#参数estimator为基模型
#参数n_features_to_select为选择的特征个数
RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
```



## 嵌入法

1. 基于惩罚项的特征选择法   通过L1正则项来选择特征：L1正则方法具有稀疏解的特性，因此天然具备特征选择的特性。

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

#带L1惩罚项的逻辑回归作为基模型的特征选择   
SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)
```

要注意，L1没有选到的特征不代表不重要，原因是两个具有高相关性的特征可能只保留了一个，如果要确定哪个特征重要应再通过L2正则方法交叉检验。



2.  基于学习模型的特征排序    这种方法的思路是直接使用你要用的机器学习算法，针对每个单独的特征和响应变量建立预测模型。假如某个特征和响应变量之间的关系是非线性的，可以用基于树的方法（决策树、随机森林）、或者扩展的线性模型等。基于树的方法比较易于使用，因为他们对非线性关系的建模比较好，并且不需要太多的调试。但要注意过拟合问题，因此树的深度最好不要太大，再就是运用交叉验证。通过这种训练对特征进行打分获得相关性后再训练最终模型。