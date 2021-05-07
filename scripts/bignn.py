import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax as scatter_softmax

import torch_scatter

from Rot3DE import Rot3DE_Trans
from rotate import RotatE_Trans
from ote import OTE 
import math

#GNN with direction
class DGNNLayer(nn.Module):
    def __init__(self, args, is_head_rel=True, **kwargs):
        super(DGNNLayer, self).__init__()
        # 具体模型：由参数确定
        self.args = args
        if args.gnn_type == "GAT":
            self.gnn_model = self.GAT_update 
        elif args.gnn_type == "GCN":
            self.gnn_model = self.GCN_update 
        else:
            raise NotImplementedError("Not implementation")

        self.is_head_rel = is_head_rel #is_head_rel==True:  update tail from head + relation
                                       #is_head_rel==False: update head from tail + relation

    # 后续具体类将会重载下面这个函数。
    # 计算实体嵌入
    # 参数：实体集合, 关系集合，边(表示为：[头实体，关系，尾实体])
    def Calculate_Entity(self, entities, relation, edge_index ):
        pos = 0 if self.is_head_rel else 2
        ent_ins = torch.index_select(entities, 0, edge_index[pos])
        return ent_ins 
    
    
    #   GCN_update & GAT_update
    # 根据是与实体相关联的所有(h,r)或者(r,t)计算实体的上下文的时候
    # 这边包括两种策略：
    # 1.基础策略：均值。GCN_update
    # 2.点乘注意力机制。GAT_update

    def GCN_update(self, ent_ins, edge_index, entities ):
        ent_pos = 2 if self.is_head_rel else 0  #ent_pos: position of ent to be upated
        # 更新torch_geometric的api接口
        # x_ents = torch.scatter(ent_ins, edge_index[ent_pos], entities.size(0), reduce="mean")
        # scatter函数发散到目的节点：mean
        x_ents = torch_scatter.scatter_mean(ent_ins, edge_index[ent_pos], dim=0, dim_size=entities.size(0))
        # x_ents = scatter_("mean", ent_ins, edge_index[ent_pos], entities.size(0))
        return x_ents 
    
    
    def GAT_update(self, ent_ins, edge_index, entities):
        def _GAT_update(k, qv, idx,  entities):
            # entities 14542 * 768
            # qv  286557 *1024
            # k 286557 *768
            # idx 286557

            embed_dim = k.size(-1)
            # f = k * qv
            alpha = (k * qv).sum(dim=-1)/math.sqrt(embed_dim)

            attn_wts = scatter_softmax(alpha.unsqueeze(1), idx, ptr=None, num_nodes=entities.size(0))
            wqv = qv*attn_wts
            # 更新torch_geometric的api接口
            out = torch_scatter.scatter_add(wqv, idx, dim=0, dim_size=entities.size(0))
            # out = scatter_("add", wqv, idx, dim_size = entities.size(0))
            return out
        ent_pos = 2 if self.is_head_rel else 0  #ent_pos: position of ent to be upated
        ent_ori = torch.index_select(entities, 0,  edge_index[ent_pos])

        zero = torch.zeros((ent_ori.size(0), self.args.hidden_dim), device=torch.device('cuda:0'), requires_grad=False)
        ent_ori = torch.cat([ent_ori, zero], dim=-1)
        x_ents = _GAT_update(ent_ori, ent_ins, edge_index[ent_pos], entities)
        return x_ents

    # general function
    # 前向传播
    def forward(self, entities, relations, edge_index):
        # compute entity embeddding
        ent_ins = self.Calculate_Entity(entities, relations, edge_index)
        # 会根据构造函数中赋值的不同，调用不同的函数: 用于计算上下文的方法：包括均值和点乘注意力。
        x_ents = self.gnn_model(ent_ins,  edge_index, entities) 
        return x_ents


class RotatEDGNNLayer(DGNNLayer):

    def __init__(self, args, dummy_node_id, embedding_range=1.0, is_head_rel=True, **kwargs):
        super(RotatEDGNNLayer, self).__init__(args, is_head_rel)
        self.embedding_range = embedding_range
        self.dummy_node_id = dummy_node_id

    def do_trans(self, ent, rel ):
        pi = 3.14159262358979323846
        phase_relation = rel/(self.embedding_range/pi)
        re_rel = torch.cos(phase_relation)
        im_rel = torch.sin(phase_relation)
        re_ent, im_ent = torch.chunk(ent, 2, dim=-1)

        # h * r
        if self.is_head_rel: #ent is head, tail-batch
            re_score = re_ent * re_rel - im_ent * im_rel
            im_score = re_ent * im_rel + im_ent * re_rel
        # t * r
        else:   #ent is tail, head-batch
            re_score = re_rel * re_ent + im_rel * im_ent
            im_score = re_rel * im_ent - im_rel * re_ent
        
        scores = torch.cat((re_score, im_score), dim=-1)
        return scores

    # ent:entity embedding; rel:index。
    def do_trans_rel_index(self, ent, rel, rel_idx):
        rel = rel.index_select(0, rel_idx)
        return self.do_trans(ent, rel)

    # eff_do_trans
    def eff_do_trans(self, ent_ids, rel_ids, entities, relation):
        def _get_abs_idx(idx1, idx2, stride):
            return idx1*stride + idx2
        def _extr_abs_idx(abs_idx, stride):
            idx1 = abs_idx // stride
            idx2 = abs_idx.fmod(stride)
            return idx1, idx2
        # entity total
        num_entity = entities.size(0)
        # relation total
        num_relation = relation.size(0)
        # <ent_idx,rel_idx> ==> abs number
        abs_idx = _get_abs_idx(ent_ids, rel_ids, num_relation)
        # 去掉重复的数字(hr)或（r.t）
        uniq_idx = torch.unique(abs_idx)
        # 定义一个实体个数 * 关系个数的Tensor。
        to_uniq_table = torch.zeros(num_entity*num_relation, device=uniq_idx.device, dtype=torch.int64)
        # 去掉重复的计算，减少计算量
        to_uniq_table[uniq_idx] = torch.arange(len(uniq_idx), device=uniq_idx.device)
        uniq_ent_ids, uniq_rel_ids = _extr_abs_idx(uniq_idx, num_relation)
        #ent_out_uniq = self.do_trans(entities.index_select(0,uniq_ent_ids), 
        #            relation.index_select(0,uniq_rel_ids))
        ent_out_uniq = self.do_trans_rel_index(entities.index_select(0, uniq_ent_ids),
                    relation, uniq_rel_ids)

        ent_out = ent_out_uniq.index_select(0, to_uniq_table[abs_idx])
        return ent_out
        
    # 对DGNNLayer中同名函数的重载。
    def Calculate_Entity(self, entities, relation, edge_index ):
        pos = 0 if self.is_head_rel else 2
        tnp = 2 if self.is_head_rel else 0
        #rel_ins = torch.index_select(relation, 0, edge_index[1])
        #ent_ins = torch.index_select(entities, 0, edge_index[pos])
        ent_out = self.eff_do_trans(edge_index[pos], edge_index[1], entities, relation)

        return ent_out 


class OTEDGNNLayer(RotatEDGNNLayer):
    def __init__(self, args, dummy_node_id, embedding_range=1.0, is_head_rel=True, **kwargs):
        super(OTEDGNNLayer, self).__init__(args, dummy_node_id, embedding_range, is_head_rel, **kwargs)
        self.ote = OTE(args.ote_size, args.ote_scale)

    def do_trans(self, ent, rel ):
        output = self.ote(ent, rel.contiguous())
        return output 

    def do_trans_rel_index(self, ent, rel, rel_idx):
        output = self.ote.forward_rel_index(ent, rel, rel_idx)
        return output
    
    def Calculate_Entity(self, entities, relation, edge_index ):
        pos = 0 if self.is_head_rel else 2
        tnp = 2 if self.is_head_rel else 0
        #rel_ins = torch.index_select(relation, 0, edge_index[1])
        #ent_ins = torch.index_select(entities, 0, edge_index[pos])
        hr_rel, tr_rel = torch.chunk(relation, 2, dim=1)
        
        relation = relation if self.is_head_rel else self.ote.orth_reverse_mat(relation)
        
        ent_out = self.eff_do_trans(edge_index[pos], edge_index[1], entities, relation)
        return ent_out


# 设置Rot3DEGNNLayer类
# 作用：用于计算知识图谱中所有实体的 <头实体关系上下文表示> 和 <关系尾实体上下文表示>。
class Rot3DEGNNLayer(DGNNLayer):

    def __init__(self, args, dummy_node_id, embedding_range=1.0, is_head_rel=True, **kwargs):
        super(Rot3DEGNNLayer, self).__init__(args, is_head_rel)
        self.embedding_range = embedding_range
        self.dummy_node_id = dummy_node_id

    # 根据KG中的(h,r)对计算h
    # 或者(r,t)对计算t
    # 提供给do_trans_rel_index函数使用。
    def do_trans(self, ent, rel):
        pi = 3.14159262358979323846
        ex, ey, ez = torch.chunk(ent, 3, dim=-1)
        rx, ry, rz, theta = torch.chunk(rel, 4, dim=-1)
        theta = theta/(self.embedding_range/pi)
        sin = torch.sin(theta)
        w = torch.cos(theta)
        rx, ry, rz = sin*rx, sin*ry, sin*rz

        # 转变成单位向量
        # --------------------------------------
        t = torch.stack([rx, ry, rz], dim=0)
        z = F.normalize(t, p=2, dim=0)
        rx, ry, rz = torch.chunk(z, 3, dim=0)
        rx = torch.squeeze(rx, dim=0)
        ry = torch.squeeze(ry, dim=0)
        rz = torch.squeeze(rz, dim=0)
        # --------------------------------------
        if self.is_head_rel:  # ent is head, tail-batch
            x_score, y_score, z_score, w_score = Rot3DE_Trans((ex, ey, ez), (rx, ry, rz, w), True)
        else:                 # ent is tail, head-batch
            x_score, y_score, z_score, w_score = Rot3DE_Trans((ex, ey, ez), (rx, ry, rz, w), False)

        scores = torch.cat((x_score, y_score, z_score, w_score), dim=-1)
        return scores

    # 参数：实体是嵌入，是关系是索引。
    # 提供给eff_do_trans函数使用
    def do_trans_rel_index(self, ent, rel, rel_idx):
        
        rel = rel.index_select(0, rel_idx)
        return self.do_trans(ent, rel)

    # eff_do_trans
    # 先去掉重复的(h,r)或者(r,t)对，以减少相同的计算。
    # 然后根据(h,r)计算t，根据(r,t)计算h
    # 提供给forword函数使用
    def eff_do_trans(self, ent_ids, rel_ids, entities, relation):
        def _get_abs_idx(idx1, idx2, stride):
            return idx1 * stride + idx2

        def _extr_abs_idx(abs_idx, stride):
            idx1 = abs_idx // stride
            idx2 = abs_idx.fmod(stride)
            return idx1, idx2

        # 实体总数
        num_entity = entities.size(0)
        # 关系总数
        num_relation = relation.size(0)
        # 实体id+关系id转变  ==> 唯一的数字id
        abs_idx = _get_abs_idx(ent_ids, rel_ids, num_relation)
        # 去复 数字id
        uniq_idx = torch.unique(abs_idx)
        # 定义一个实体总数 * 关系总数的tensor向量。
        to_uniq_table = torch.zeros(num_entity * num_relation, device=uniq_idx.device, dtype=torch.int64)
        # 在数组中记录独一无二的[] 中存储：
        to_uniq_table[uniq_idx] = torch.arange(len(uniq_idx), device=uniq_idx.device)
        uniq_ent_ids, uniq_rel_ids = _extr_abs_idx(uniq_idx, num_relation)
        # ent_out_uniq = self.do_trans(entities.index_select(0,uniq_ent_ids),
        #            relation.index_select(0,uniq_rel_ids))
 
        # 返回计算结果
        ent_out_uniq = self.do_trans_rel_index(entities.index_select(0, uniq_ent_ids),
                                               relation, uniq_rel_ids)   
        ent_out = ent_out_uniq.index_select(0, to_uniq_table[abs_idx])
        return ent_out

    # 对DGNNLayer中同名函数的重载。
    # 功能：已知(h,r)计算t 或者 已知(r,t)计算h;
    def Calculate_Entity(self, entities, relation, edge_index):
        pos = 0 if self.is_head_rel else 2
        tnp = 2 if self.is_head_rel else 0
        # rel_ins = torch.index_select(relation, 0, edge_index[1])
        # ent_ins = torch.index_select(entities, 0, edge_index[pos])
        ent_out = self.eff_do_trans(edge_index[pos], edge_index[1], entities, relation)

        return ent_out


# BiGNN是程序的基础框架类
# 设置：在BiGNN内部设置Rot3DE模块(对应于文中Rot3DE模型)。
class BiGNN(nn.Module):
    def __init__(self,  num_entity, num_rels,  args, embedding_range=1.0, model="RotatE"):  # default RotatE
        super(BiGNN, self).__init__()
        self.num_entity = num_entity
        self.num_rels = num_rels

        # 设置内部具体模型：
        if model == 'RotatE':
            self.head_rel = RotatEDGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=True)
            self.tail_rel = RotatEDGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=False)
        elif model == 'OTE':
            self.head_rel = OTEDGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=True)
            self.tail_rel = OTEDGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=False)
        elif model == "Rot3DE":
            self.head_rel = Rot3DEGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=True)
            self.tail_rel = Rot3DEGNNLayer(args, self.dummy_ent_id, embedding_range=embedding_range, is_head_rel=False)
        else:
            raise ValueError("Not defined!")

    @property
    def dummy_ent_id(self):
        return self.num_entity - 1
    @property
    def dummy_rel_id(self):
        return self.num_rels - 1

    def retrival_emb(self, ent_emb, ent_id, rel_id, is_hr=True):
        if ent_id.dim() == 1:
            return ent_emb.index_select(0, ent_id).unsqueeze(1)
        bsz, neg_sz = ent_id.size()
        return ent_emb.index_select(0, ent_id.view(-1)).view(bsz, neg_sz, -1)

    def forward(self, entities, relation,  edge_index):
        # calculate hr
        # add dummy_head
        edge_dummy_idx = [ torch.LongTensor([self.dummy_ent_id, self.dummy_rel_id, i]) for i in range(self.num_entity) ] 
        edge_dummy_idx = torch.stack(edge_dummy_idx).transpose(0,1).to(edge_index.device) #3 X N
        # enhance
        edge_aug_idx = torch.cat((edge_index, edge_dummy_idx), dim=1)
        # 返回hr表示
        hr = self.head_rel(entities, relation, edge_aug_idx)
        # calculate tr
        # add dummy tail
        # enhance
        edge_dummy_idx = [ torch.LongTensor([i, self.dummy_rel_id, self.dummy_ent_id]) for i in range(self.num_entity) ] 
        edge_dummy_idx = torch.stack(edge_dummy_idx).transpose(0,1).to(edge_index.device) #3 X N
        edge_aug_idx = torch.cat((edge_index, edge_dummy_idx), dim=1)
        # 返回rt表示
        tr = self.tail_rel(entities, relation, edge_aug_idx)
        return hr, tr
