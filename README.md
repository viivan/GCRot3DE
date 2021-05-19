# Knowledge-graph-representation-learning-model-GCRot3DE

本项目由三个文件夹构成models，scripts，data分别用于存储模型，代码实现和数据集。

## 1.GCRot3DE模型

(1).设置quaternion_product，forword_rotate , backword_rotate，Rotat3DE_Trans函数，分别提供了四元数乘法运算，从头实体到尾实体的三维旋转变换(记为正向旋转)，从尾实体到头实体的三维旋转变换(记为反向旋转)，Rotat3DE_Trans则为正向、反向旋转变换提供了统一结构。

> 详细可见scripts/Rot3DE.py

(2).设置Rot3DEGNNLayer类，用于自适应感知实体在图中重要的领域信息，捕获实体的图上下文表示。为知识图谱中所有实体，计算两种实体图上下文表示 : 头实体-关系上下文表示(HRCR) 和 关系-尾实体上下文表示(RTCR)。

> 详细可见scripts/bignn.py

(3).设置Rot3DE函数，是三维旋转模型的具体实现，即头实体h通过关系r中定义的三维旋转操作之后 与 尾实体h的距离(对应于文中的Rot3DE模型，中间模型) ，其距离打分函数(Score function)如下：

​                                                                           $d((h,r),t) = ||h * r - t||$

> 详细可见scripts/model.py

(4).设置Rot3DED函数，是三维旋转模型的双向旋转版本，其距离打分函数如下：

​                     $d(h,r,t)= d((h,r),t) + d(h,(r,t)) = ||r^{-1}* h * r - t|| + ||r* t * r^{-1} - h||$

(5).设置GCRot3DE函数，是融合实体图上下文的三维旋转知识图谱表示学习模型的具体实现(对应于文中的GCRot3DE模型），模型的距离打分函数定位为:

​                                     $$d(h,r,t) = (d((h,r),t) + d_c(h,(r,t)) + d((h,r),t) + d_c(h,(r,t)) ) / 4 $$


> 详细可见scripts/model.py，scripts/bignn.py.

(6).其他脚本设置：

 设置data/FB15k-237/n-n-classify.py脚本。该脚本根据每个关系对应头实体的平均数目和尾实体的平均数目，将关系进行归类为四种类型，包括1-1，1-n, n-1和n-n。对于测试集中的每个三元组，按三元组中关系的具体类型，将测试三元组进行划分；测试集中的每个三元组将会被划分到四个文件中：1-1.txt， 1-n.txt，  n-1.txt 和  n-n.txt。（脚本主要用于FB15k-237数据集）

设置data/wn18rr/divide-test-set-by-relation-name.py脚本。该脚本按照关系名称对数据集中的测试集按照关系具体名称进行划分。（主要用于WN18RR数据集，WN18RR中只包含10多种关系，按照关系具体名称，将测试数据集中的三元组划分为10多个文件，每个文件只包含一种关系）

## 2.模型训练步骤:

- 预训练Rot3DE模型。
- 训练融合实体图上下文的GCRot3DE模型。

## 3.模型参数设置

- 模型在FB15k-237 数据集上的参数设置。

  (1).预训练Rot3DE.	
  
  ```shell
  python -u scripts/run.py --do_train --cuda --do_valid --do_test --data_path /root/knowledge-graph-representation-learning-GCRot3DE/data/FB15k-237 --model Rot3DE -n 256 -b 512 -d 256  -g 15.0 -a 0.5 -adv -lr 0.002  --max_steps 250000  --test_batch_size  16  --three_entity_embedding  --four_relation_embedding   --log_steps 1000 --valid_steps 10000 --schedule_steps 25000  -save /root/knowledge-graph-representation-learning-GCRot3DE/models/Rot3DE_FB15k-237
  ```
  
   (2).在预训练模型的基础上，训练GCRot3DE.
  
  ```shell
  python -u scripts/run.py --do_train --cuda  --do_valid  --do_test --data_path  /root/knowledge-graph-representation-learning-GCRot3DE/data/FB15k-237 --model GCRot3DE -n 256 -b 128 -d 256   -g 12.0 -a 1.5 -adv  -lr 0.0002    --three_entity_embedding  --four_relation_embedding   --max_steps 100000 --test_batch_size  16  --log_steps 1000 --valid_steps 5000  --schedule_steps 8000  --add_dummy --test_split_num 10 --init_embedding  /root/knowledge-graph-representation-learning-GCRot3DE/models/Rot3DE_FB15k-237/best  -save /root/knowledge-graph-representation-learning-GCRot3DE/models/GCRot3DE_FB15k-237
  ```
  
  


- 模型在WN18RR数据集上的参数设置：

    (1).预训练Rot3DE.
    
    ```shell
    python -u scripts/run.py --do_train --cuda  --do_valid  --do_test --data_path /home/mk/knowledge-graph-representation-learning-GCRot3DE/data/wn18rr  --model Rot3DE -n 256 -b 512 -d 256  -g 5.0 -a 1.5 -adv -lr 0.001  --max_steps 250000 --test_batch_size  16   --three_entity_embedding  --four_relation_embedding --log_steps 1000 --valid_steps 10000  --schedule_steps 25000  -save /home/mk/knowledge-graph-representation-learning-GCRot3DE/models/Rot3DE_wn18rr
    ```
    
    (2).在预训练模型的基础上，训练GCRot3DE.

    ```shell
    python -u scripts/run.py --do_train --cuda  --do_valid  --do_test --data_path  /home/mk/knowledge-graph-representation-learning-GCRot3DE/data/FB15k-237 --model GCRot3DE -n 256 -b 128 -d 256   -g 10.0 -a 0.5 -adv  -lr 0.00003    --three_entity_embedding  --four_relation_embedding   --max_steps 100000 --test_batch_size  16  --log_steps 1000 --valid_steps 5000  --schedule_steps 8000  --add_dummy --test_split_num 10 --init_embedding  /home/mk/knowledge-graph-representation-learning-GCRot3DE/models/Rot3DE_wn18rr/best  -save /home/mk/knowledge-graph-embedding-CRot3DE/models/GCRot3DE_wn18rr
    ```

  
