# 读取实体
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


lef = {}
rig = {}
rellef = {}
relrig = {}

triple = open("train.txt", "r")
valid = open("valid.txt", "r")
test = open("test.txt", "r")

# 读取 训练集
# tot = 272115
for content in triple:
	# content = triple.readline()
	h,r,t = content.strip().split()
	h,t,r = entity2id[h], entity2id[t], relation2id[r]
	if not (h,r) in lef:
		lef[(h,r)] = []    
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)  # 加入以(h,r) 开头的尾实体[] lef
	rig[(r,t)].append(h)  # 加入以(r,t) 开头的头实体[] rig
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	# 
	rellef[r][h] = 1 # 关系左侧，存在h  
	relrig[r][t] = 1 # 关系右侧，存在t

# 读取验证集
# tot = 17535   #fb15k-237
# for i in range(tot):
#	content = valid.readline()
for content in valid:
	h,r,t = content.strip().split()
	h, t, r = entity2id[h], entity2id[t], relation2id[r]
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

# 读取测试集
# tot = 20466
# for i in range(tot):
# 	content = test.readline()
for content in test:
	h,r,t = content.strip().split()
	h, t, r = entity2id[h], entity2id[t], relation2id[r]
	if not (h,r) in lef:
		lef[(h,r)] = []
	if not (r,t) in rig:
		rig[(r,t)] = []
	lef[(h,r)].append(t)
	rig[(r,t)].append(h)
	if not r in rellef:
		rellef[r] = {}
	if not r in relrig:
		relrig[r] = {}
	rellef[r][h] = 1
	relrig[r][t] = 1

test.close()
valid.close()
triple.close()

# # 读取 类型限制
# f = open("type_constrain.txt", "w")
# f.write("%d\n"%(len(rellef)))    # 总的限制数;
# # 遍历所有的 关系
# for i in rellef:
# 	# 打印 关系id, 右侧实体种类数
# 	f.write("%s\t%d"%(i,len(rellef[i])))
# 	# 打印 右侧实体种类
# 	for j in rellef[i]:
# 		f.write("\t%s"%(j))
# 	f.write("\n")
# 	# 换行
# 	# 打印右侧 实体
# 	f.write("%s\t%d"%(i,len(relrig[i])))
# 	for j in relrig[i]:
# 		f.write("\t%s"%(j))
# 	f.write("\n")
# f.close()



rellef = {}
totlef = {}
relrig = {}
totrig = {}
# lef: (h, r)
# rig: (r, t)
# 遍历所有(h,r)组合
for i in lef:
	# 关系r：不在rellef中
	if not i[1] in rellef:
		rellef[i[1]] = 0
		totlef[i[1]] = 0
	rellef[i[1]] += len(lef[i])  # 统计每个关系r有多少右实体；
	totlef[i[1]] += 1.0		     # 统计每个关系r有多少不同左实体；

# 遍历所有（r,t）
for i in rig:
	# 关系r: 不在rellef中
	if not i[0] in relrig:
		relrig[i[0]] = 0
		totrig[i[0]] = 0
	relrig[i[0]] += len(rig[i])
	totrig[i[0]] += 1.0

s11=0
s1n=0
sn1=0
snn=0
f = open("test.txt", "r")
tot = 20466  # fb15k-237

# 默认1对1情况下： 头实体种类 = 尾实体个数（也是种类数是一个）；
# 读取测试集数据
# for i in range(tot):
# 	content = f.readline()
for content in f:
	h,r,t = content.strip().split()
	h, t, r = entity2id[h], entity2id[t], relation2id[r]
	rign = rellef[r] / totlef[r]
	lefn = relrig[r] / totrig[r]

	if (rign < 1.5 and lefn < 1.5):
		s11+=1
	if (rign >= 1.5 and lefn < 1.5):
		s1n+=1
	if (rign < 1.5 and lefn >= 1.5):
		sn1+=1
	if (rign >= 1.5 and lefn >= 1.5):
		snn+=1
f.close()


f = open("./test.txt", "r")
f11 = open("./temp/1-1.txt", "w")
f1n = open("./temp/1-n.txt", "w")
fn1 = open("./temp/n-1.txt", "w")
fnn = open("./temp/n-n.txt", "w")
fall = open("./temp/test_all.txt", "w")

# tot = 20466
# fall.write("%d\n"%(tot))
# f11.write("%d\n"%(s11))
# f1n.write("%d\n"%(s1n))
# fn1.write("%d\n"%(sn1))
# fnn.write("%d\n"%(snn))

# print("1111111111111")
#   
# for i in range(tot):
# 	content = f.readline()
RotatE3D

for content in f:
	h,r,t = content.strip().split()
	h, t, r = entity2id[h], entity2id[t], relation2id[r]
	rign = rellef[r] / totlef[r] # 尾实体个数  / 头实体种类
	lefn = relrig[r] / totrig[r] # 头实体个数 / 尾实体种类  
	# 判断当前关系 类型是：1对1， 1对n， n对1， n对n 分别放置到 不同的位置;
	# print(content)
	if (rign < 1.5 and lefn < 1.5):
		f11.write(content)
		fall.write("0"+"\t"+content)
	if (rign >= 1.5 and lefn < 1.5):
		f1n.write(content)
		fall.write("1"+"\t"+content)
	if (rign < 1.5 and lefn >= 1.5):
		fn1.write(content)
		fall.write("2"+"\t"+content)
	if (rign >= 1.5 and lefn >= 1.5):
		fnn.write(content)
		fall.write("3"+"\t"+content)

fall.close()
f.close()
f11.close() # 192
f1n.close() # 1293
fn1.close() # 4185
fnn.close() # 14796

print("over")
