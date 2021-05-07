#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @date: 2021/12/14 21:17
# @file: Rot3DE
# @author：Sidewinder
import torch
# 设置了头实体关系到尾实体的转换
# 即正向(三维)旋转：由(头实体和关系)映射获得尾实体
# (h,r)=>t
def fun1(h, r):
    hx, hy, hz = h
    rx, ry, rz, rw = r
    x1, y1, z1, w1 = mult((rx, ry, rz, rw), (hx, hy, hz, 0))
    x1, y1, z1, w1 = mult((x1, y1, z1, w1), (-rx, -ry, -rz, w1))
    return x1, y1, z1, w1

# 设置了关系尾实体到头实体的转换
# 即反向(三维)旋转：由(尾实体和关系)映射获得头实体
# (r,t)=>h
def fun2(r, t):
    tx, ty, tz = t
    rx, ry, rz, rw = r
    x1, y1, z1, w1 = mult((-rx, -ry, -rz, rw), (tx, ty, tz, 0))
    x1, y1, z1, w1 = mult((x1, y1, z1, w1), (rx, ry, rz, w1))

    return x1, y1, z1, w1

# 设置四元数乘法运算
def mult(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return x3, y3, z3, w3

# 设置统一转换 
# 参数：( 实体，关系，是否为hr)
# 返回转换之后结果
# return ent
def Rot3DE_Trans(ent, rel, is_hr):
    if is_hr:   #ent == head
        x, y, z, w = fun1(ent, rel)
    else:   #ent == tail
        x, y, z, w = fun2(rel, ent)
    return x, y, z, w
