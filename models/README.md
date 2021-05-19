## 文件说明

models文件夹初始状态为空，用于存储模型训练之后的结果。

每个模型文件夹或将包括有以下文件：

| 文件                   | 说明                                                         |
| :--------------------- | ------------------------------------------------------------ |
| checkpoint             | save_checkpoint_steps 阶段性存储模型。                       |
| config.json            | 模型的参数配置                                               |
| entity_embedding.npy   | 以numpy文件的形式，存储实体嵌入矩阵。                        |
| relation_embedding.npy | 以numpy文件的形式，存储关系嵌入矩阵。                        |
| train.log              | 训练日志                                                     |
| test.log               | 测试日志                                                     |
| best文件夹             | 存放模型训练阶段，MRR指标表现最好的模型参数(文件夹中或将包括有checkpoint，config.json，entity_embedding.npy，relation_embedding.npy 等文件) |