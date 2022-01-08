1. 对数据进行清理
2. 随机划分数据为训练集，验证集和测试集，也可以采用k-cross验证
3. 观察训练集，验证集和测试集分布是否一致
4. 进行特征工程（反复进行）
5. 对数据进行归一化，深度学习可以用batch normalization
6. 用简单模型（如svm，多层感知机）对任务求解，作为baseline
5. 训练备选的模型
   * 训练集结果不好
     * 结果超过baseline：
       * 加特征
       * 更复杂的模型
     * 结果未超过baseline：
       * 优化方法有问题，尝试：
         1. SGD
         2. 自适应学习率，warm up
         3. 更换激活函数
         4. batch normalization
   * 训练集结果很好
     * 测试集结果不好：
       * 过拟合，尝试：
         1. 增大数据量
         2. 数据增广
         3. 减少特征
         4. 降低模型复杂度（减少神经元数量）
         5. 正则化
         6. dropout
         7. 早停
       * mismatch:
         1. 换数据
         2. 迁移学习
     * 测试集结果很好：
       * 进一步优化
         * bad case分析
         * 迭代特征工程
       * 结果满意
         * 下班