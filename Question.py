# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:50:48 2024
function!

@author: wu22
"""
import numpy as np

class MultiObjectiveOptimization:
    def __init__(self, x1_max=10, x2_max=10):
        """
        初始化 MultiObjectiveOptimization 类的实例，设置默认参数值。

        参数:
        x1_max (float): 自变量 x1 的最大限制。默认值为 10。
        x2_max (float): 自变量 x2 的最大限制。默认值为 10。
        """
        self.x1_max = x1_max
        self.x2_max = x2_max

    def objective1(self, x):
        """目标函数1：最大化总收益"""
        x1, x2 = x
        return -(x1 + x2)  # 负号因为要最大化

    def objective2(self, x):
        """目标函数2：最小化总成本"""
        x1, x2 = x
        return 2 * x1**2 + 3 * x2**2 + x1 + x2

    def constraint1(self, x):
        """约束条件1：非线性资源限制，返回布尔值"""
        x1, x2 = x
        return (x1 - 2)**2 + (x2 - 2)**2 <= 16  # 检查是否满足圆形区域限制

    def constraint2(self, x):
        """约束条件2：非负性，返回布尔值"""
        x1, x2 = x
        return x1 >= 0 and x2 >= 0  # 检查 x1 和 x2 是否为非负值

