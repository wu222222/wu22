# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:50:48 2024
function!

@author: wu22
"""
import numpy as np

class CTP1:
    def __init__(self, n):
        """
        初始化 CTP1 问题的参数
        :param n: 设计变量的数量
        """
        self.n = n
        self.a_j = [0.858, 0.728]  # 第一个和第二个约束的 a_j 参数
        self.b_j = [0.541, 0.295]  # 第一个和第二个约束的 b_j 参数

    def f1(self, x):
        """
        计算第一个目标函数 f1
        :param x: 设计变量向量
        :return: f1 的值
        """
        return x[0]

    def g(self, x):
        """
        计算辅助函数 g
        :param x: 设计变量向量
        :return: g 的值
        """
        x = np.array(x)
        return 1 + 100 * np.sum((x[1:] - 0.5)**2)

    def f2(self, x):
        """
        计算第二个目标函数 f2
        :param x: 设计变量向量
        :return: f2 的值
        """
        g_value = self.g(x)
        return g_value * np.exp(-self.f1(x) / g_value)

    def constraints(self, x):
        """
        计算约束条件，返回布尔值表示是否满足约束
        :param x: 设计变量向量
        :return: 布尔数组，表示各个约束是否满足
        """
        f1_value = self.f1(x)
        f2_value = self.f2(x)
        constraints_satisfied = []
        for a, b in zip(self.a_j, self.b_j):
            constraint_value = f2_value - a * np.exp(-b * f1_value)
            constraints_satisfied.append(constraint_value >= 0)
        return np.array(constraints_satisfied).all()
    
    def constraints_func(self, x):
        """
        计算约束条件，返回布尔值表示是否满足约束
        :param x: 设计变量向量
        :return: 布尔数组，表示各个约束是否满足
        """
        f1_value = x[0]
        f2_value = x[1]
        constraints_satisfied = []
        for a, b in zip(self.a_j, self.b_j):
            constraint_value = f2_value - a * np.exp(-b * f1_value)
            constraints_satisfied.append(constraint_value >= 0)
        return np.array(constraints_satisfied).all()

