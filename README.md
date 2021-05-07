# GCRot3DE

本项目由三个文件夹构成models，scripts，data分别用于存储模型，代码实现和数据集。

## 1.CRot3DE模型

(1).设置Rot3DE.py文件，内部提供了四元数乘法运算, 从头实体到尾实体的三维旋转变换(记为正向旋转)，从尾实体到头实体的三维旋转变换(记为反向旋转)。

(2).设置Rot3DEGNNLayer类，用于引入实体的上下文，即用于计算知识图谱中所有实体的 <头实体关系上下文表示> 和 <关系尾实体上下文表示>。

(3).设置Rot3DE函数，是三维旋转模型的具体实现，即头实体h通过关系r中定义的三维旋转操作之后 与 尾实体h的距离(对应于文中的Rot3DE模型) 。

​                                                                           $d((h,r),t) = ||h * r - t||$

(4).设置Rot3DED(three-dimensional rotate embedding with double directions)函数，是三维旋转模型的双向旋转版本。

​                     $d(h,r,t)= d((h,r),t) + d(h,(r,t)) = ||r^{-1}* h * r - t|| + ||r* t * r^{-1} - h||$

(5).设置CRot3DE函数，引入实体上下文的三维旋转嵌入模型具体实现(对应于文中的CRot3DE模型）。 

用于计算引入上下文的CRot3DE中的距离:

​                                     $$d(h,r,t) = (d((h,r),t) + d_c(h,(r,t)) + d((h,r),t) + d_c(h,(r,t)) ) / 4 $$



> 详细可见scripts/model.py，scripts/bignn.py.

(6).其他脚本设置：

(1).设置data/FB15k-237/n-n-classify.py脚本。该脚本根据每个关系对应头实体的平均数目和尾实体的平均数目，将关系进行归类一下四种类型，包括1-1，1-n, n-1和n-n。首先， 将测试集中的每个三元组，按三元组中关系的具体类型，将测试三元组进行划分；然后，测试集中的三元组将会被划分到四个文件中：1-1.txt， 1-n.txt，  n-1.txt 和  n-n.txt。（主要用于FB15k-237数据集）

(2).设置data/wn18rr/divide-test-set-by-relation-name.py脚本。该脚本按照关系名称对数据集中的测试集按照关系具体名称进行划分。（主要用于WN18RR数据集，WN18RR中只包含10多种关系，按照关系具体名称，将测试三元组划分为10多个文件，每个文件只包含一种关系）

## 2.模型训练步骤:

- 预训练Rot3DE模型。
- 训练引入实体上下文的CRot3DE模型。

## 3.模型参数设置

- 模型在FB15k-237 数据集上的参数设置。

  (1).预训练Rot3DE.

```shell
python -u scripts/run.py --do_train --cuda --do_valid --do_test --data_path /home/mk/knowledge-graph-embedding-CRot3DE/data/FB15k-237 --model Rot3DE -n 256 -b 512 -d 256  -g 15.0 -a 0.5 -adv -lr 0.002  --max_steps 250000  --test_batch_size  16  --three_entity_embedding  --four_relation_embedding   --log_steps 10000 --valid_steps 10000 --schedule_steps 25000  -save /home/mk/knowledge-graph-embedding-CRot3DE/models/Rot3DE_FB15k-237
```

​	  (2).在预训练模型的基础上，训练CRot3DE.

```shell
python -u scripts/run.py --do_train --cuda  --do_valid  --do_test --data_path  /home/mk/knowledge-graph-embedding-CRot3DE/data/FB15k-237 --model CRot3DE -n 256 -b 100 -d 256   -g 12.0 -a 1.4 -adv  -lr 0.0002    --three_entity_embedding  --four_relation_embedding   --max_steps 100000 --test_batch_size  16  --log_steps 1000 --valid_steps 5000  --schedule_steps 8000  --add_dummy --test_split_num 10 --init_embedding  /home/mk/knowledge-graph-embedding-CRot3DE/models/Rot3DE_FB15k-237/best  -save /home/mk/knowledge-graph-embedding-CRot3DE/models/CRot3DE_FB15k-237
```

- 模型在WN18RR数据集上的参数设置：

  (1).预训练Rot3DE.

  ```shell
  python -u scripts/run.py --do_train --cuda  --do_valid  --do_test --data_path /home/mk/knowledge-graph-embedding-CRot3DE/data/wn18rr  --model Rot3DE -n 256 -b 512 -d 256  -g 5.0 -a 1.8 -adv -lr 0.001  --max_steps 250000 --test_batch_size  16   --three_entity_embedding  --four_relation_embedding --log_steps 1000 --valid_steps 10000  --schedule_steps 25000  -save /home/mk/knowledge-graph-embedding-CRot3DE/models/Rot3DE_wn18rr
  ```

  (2).在预训练模型的基础上，训练CRot3DE.

  ```shell
  python -u scripts/run.py --do_train --cuda  --do_valid  --do_test --data_path  /home/mk/knowledge-graph-embedding-CRot3DE/data/FB15k-237 --model CRot3DE -n 256 -b 100 -d 256   -g 10.0 -a 0.5 -adv  -lr 0.00003    --three_entity_embedding  --four_relation_embedding   --max_steps 100000 --test_batch_size  16  --log_steps 1000 --valid_steps 5000  --schedule_steps 8000  --add_dummy --test_split_num 10 --init_embedding  /home/mk/knowledge-graph-embedding-CRot3DE/models/Rot3DE_FB15k-237/best  -save /home/mk/knowledge-graph-embedding-CRot3DE/models/CRot3DE_FB15k-237
  ```

  



  

