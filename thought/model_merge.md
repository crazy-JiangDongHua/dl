## 模型融合简介

#### 什么是模型融合

一般来说，通过融合多个不同的模型，可能提升机器学习的性能，这一方法在各种机器学习比赛中广泛应用，比如在kaggle上的otto产品分类挑战赛①中取得冠军和亚军成绩的模型都是融合了1000+模型的“庞然大物”。常见的集成学习&模型融合方法包括：简单的Voting/Averaging（分别对于分类和回归问题）、Stacking、Boosting和Bagging。

#### 模型融合应用广泛的原因

可以通过数学证明模型，随着集成中个体分类器数目T 的增大，集成的**错误率将指数级下降**，最终趋向于零。具体证明在周志华和李航老师的书中都有。它不是具体的指某一个算法，而是一种把多个弱模型融合合并在一起变成一个强模型的思想,也称为集成学习(Ensemble Learning)：

1. 单个模型容易过拟合，多个模型融合可以提高范化能力
2. 单个模型预测能力不高，多个模型往往能提高预测能力
3. 对于数据集过大或过小，可以分别进行划分和有放回的操作，产生不同的数据子集，然后通过数据子集训练不同的分类模型，最终合并成一个大的分类器
4. 对于多个异构的特征集的时候，很难进行融合，可以考虑每个数据集构建一个分类模型，然后将多个模型融合
5. 模型融合算法成功的关键在于能保证弱分类器（弱模型）的多样性，融合不稳定的学习算法能得到更明显的性能提升

#### 模型融合的条件

个体学习器准确性越高、多样性越大，则融合越好。（E = E'-A'，E代表融合后的误差加权均值，E'代表个体学习器的误差加权均值，A'代表模型的多样性，也就是各个模型之间的分歧值）

Base Model 之间的相关性要尽可能的小。这就是为什么非 Tree-based Model 往往表现不是最好但还是要将它们包括在 Ensemble 里面的原因。Ensemble 的 Diversity 越大，最终 Model 的 Bias 就越低。
Base Model 之间的性能表现不能差距太大。这其实是一个 Trade-off，在实际中很有可能表现相近的 Model 只有寥寥几个而且它们之间相关性还不低。但是实践告诉我们即使在这种情况下 Ensemble 还是能大幅提高成绩。

#### 模型融合的分类

按照个体学习器的关系可以分为两类:

- 个体学习器问存在强依赖关系、必须串行生成的序列化方法，代表Boosting方法；
- 个体学习器间不存在强依赖关系、可同时生成的并行化方法，代表是Bagging 和”随机森林” 。



## 模型融合详细介绍

#### 模型融合基础算法

1. 投票法（Voting）：如果是分类模型，每个模型都会给出一个类别预测结果，通过投票的方式，按照少数服从多数的原则融合得到一个新的预测结果。
2. 均值法（Averaging）：如果是回归模型，每个模型给出的预测结果都是数值型的，这时候我们可以通过求所有子模型的预测结果的均值作为最终的融合结果。
3. 加权平均：对均值法的改进，根据模型的表现给予不同的权重。例如模型A，B，C的表现排名为1，2，3，那么相对应的权重为3/6，2/6，1/6

#### 个体学习器依赖的融合-boosting（串行）

Boosting是一种将各种弱分类器串联起来的集成学习方式，每一个分类器的训练都依赖于前一个分类器的结果，顺序运行的方式导致了运行速度慢。和所有融合方式一样，它不会考虑各个弱分类器模型本身结构为何，而是对训练数据（样本集）和连接方式进行操纵以获得更小的误差。但是为了将最终的强分类器的误差均衡，之前所选取的分类器一般都是相对比较弱的分类器，因为一旦某个分类器较强将使得后续结果受到影响太大。

常见的Boosting方法有Adaboost、GBDT、XGBOOST等

这里不详细介绍。

#### 个体学习器不依赖的融合-bagging（并行）

Bagging就是采用有放回的方式进行抽样，用抽样的样本建立子模型,对子模型进行训练，这个过程重复多次，最后进行融合。

主要的算法有bagging和随机森林。

随机森林的主要优点是：

1. 具有极高的准确率
2. 随机性的引入，使得随机森林不容易过拟合
3. 随机性的引入，使得随机森林有很好的抗噪声能力
4. 能处理很高维度的数据，并且不用做特征选择
5. 既能处理离散型数据，也能处理连续型数据，数据集无需规范化
6. 训练速度快，可以得到变量重要性排序
7. 容易实现并行化

随机森林的主要缺点是：

1. 当随机森林中的决策树个数很多时，训练时需要的空间和时间会较大
2. 随机森林模型还有许多不好解释的地方，有点算个黑盒模型

####  Stacking——各种机器学习比赛中被誉为“七头龙神技”

但因其模型的庞大程度与效果的提升程度往往不成正比，所以一般很难应用于实际生产中。

下面以一种易于理解但不会实际使用的两层的stacking方法为例，简要说明其结构和工作原理：（这种模型问题将在后续说明）假设我们有三个基模型M1,M2,M3，用训练集对其进行训练后，分别用来预测训练集和测试集的结果，得到P1，T1，P2，T2，P3，T3。我们将P1,P2,P3合并，作为下一层的训练集，用新的训练集训练模型M4。然后用M4来预测新的测试集（T1,T2,T3合并）得到最终的预测结果。

**注意：对不同的model可以采用不同的特征**

 这种方法的问题在于，模型M1/2/3是我们用整个训练集训练出来的，我们又用这些模型来预测整个训练集的结果，毫无疑问过拟合将会非常严重。

因此在实际应用中往往采用交叉验证的方法来解决过拟合问题。

以5折交叉验证为例：

1. 首先我们将训练集分为五份。
2. 对于每一个基模型来说，我们用其中的四份来训练，然后对未用来的训练的一份训练集和验证集进行预测。然后改变所选的用来训练的训练集和用来验证的训练集，重复此步骤，直到获得完整的训练集的预测结果。
3. 对五个模型，分别进行步骤2，我们将获得5个模型，以及五个模型分别通过交叉验证获得的验证集预测结果。即P1、P2、P3、P4、P5。
4. 用五个模型分别对测试集进行预测，得到测试集的预测结果：T1、T2、T3、T4、T5。
5. 将**P1~5合并、T1~5取平均**作为下一层的训练集和测试集。

这样实现就是两层循环，第一层是模型，第二层是k折交叉验证。**第二层的模型一般为了防止过拟合会采用简单的模型。还有一种思想是：第二层的输入数据，除了第一层的训练结果外，还包括了原始特征。**示意图如下：

![img](./stacking.png)

#### Blending----与Stacking类似，只是由K-FoldCV 改成 HoldOutCV

Blending是一种和Stacking很相像的模型融合方式，它与Stacking的区别在于训练集不是通过K-Fold的CV策略来获得预测值从而生成第二阶段模型的特征，而是建立一个Holdout集，例如10%的训练数据，第二阶段的stacker模型就基于第一阶段模型对这10%训练数据的预测值进行拟合。说白了，就是把Stacking流程中的K-Fold CV 改成HoldOut CV。

Stacking与Blending相比，Blending的优势在于：

1. Blending比较简单，而Stacking相对比较复杂；
2. 能够防止信息泄露：generalizers和stackers使用不同的数据； 
3. 不需要和你的队友分享你的随机种子；

而缺点在于：
1. 只用了整体数据的一部分； 
2. 最终模型可能对留出集（holdout set）过拟合；
3. Stacking多次交叉验证要更加稳健。
