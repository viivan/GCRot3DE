#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @date: 2021/1/15 14:47
# @file: 根据关系划分数据集
# @author：Sidewinder


# 目前的目标是：
# 1.将清华那边的那个1-1,1-n,n-1, n-n 移植到这个项目里面。（这个修改起来应该也不是很难的）(搞定了)
# 2.写一个关系具体的名字划分数据的方案。（这个代码也简单的）
# 3.还有一个就是将四元数旋转的模型移动过来可以考虑。
# 4.看一下GNN的代码。

import os
print(os.getcwd())
with open(os.path.join('.', 'entities.dict')) as fin:
	entity2id = dict()
	id2entity = dict()
	for line in fin:
		eid, entity = line.strip().split('\t')
		entity2id[entity] = int(eid)
		id2entity[int(eid)] = entity

# 读取关系
with open(os.path.join('.', 'relations.dict')) as fin:
	relation2id = dict()
	id2relation = dict()
	for line in fin:
		rid, relation = line.strip().split('\t')
		relation2id[relation] = int(rid)
		id2relation[int(rid)] = relation


test = open("test.txt", "r")
rid2triple = dict()
count = 0
for line in test:
	count += 1
	h, r, t = line.strip().split()
	h, t, r = entity2id[h], entity2id[t], relation2id[r]
	print(h, r, t)
	if r not in rid2triple:
		rid2triple[r] = []
	rid2triple[r].append(line)
print(count)
test.close()
count = 1
for rid in rid2triple:
	print(id2relation[rid] + "test.txt")

	rel_file = open("temp/"+str(count) + "_test.txt", "w")
	for content in rid2triple[rid]:
		rel_file.write(content)
	rel_file.close()
	count += 1
