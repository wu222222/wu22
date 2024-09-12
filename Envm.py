import numpy as np
import math
from Question import MultiObjectiveOptimization

class Env():
    def __init__(self,actions,fresh_time=0.3,popular_size=100,precision=5):
        self.FRESH_TIME = fresh_time
        self.pop = popular_size
        
        self.func = MultiObjectiveOptimization()
        self.xp = [5,5]
        self.xd = [-5,-5]
        self.dim = len(self.xp)
        
        self.F = np.round(np.random.uniform(size=self.pop),precision)
        self.CR = np.round(np.random.uniform(size=self.pop),precision)
        
        self.X = np.zeros((self.pop,self.dim))
        for j in range(self.pop):
            for i in range(self.dim):
                self.X[j][i] = round(self.xd[i] + (self.xp[i] - self.xd[i]) * np.random.uniform(),precision) 
        self.actions = actions
        self.result = []
        
        self.pltn = self.pop * 2
        self.pltx = []       
        self.elitesN = math.floor(self.pop)
        

    def obj_fun(self,x):
        return np.sum(self.base_obj_fun(x))
        
    def obj_fun2(self,x1,x2):
        return self.base_obj_fun([x1,x2])
    
    def base_obj_fun(self,x):
        f1 = x[0]**4 - 10 * x[0] ** 2 + x[0] * x[1] + x[1] ** 4 - x[0]**2 * x[1] ** 2
        f2 = x[1]**4 - x[0]**2 * x[1] ** 2 + x[0] ** 4 + x[0] * x[1]
        return f1,f2
    
    def get_env_feedback(self,i,action):
            self.F[i] += action[0]
            self.CR[i] += action[1]

    def update_env(s,episode,step_counter):
        return s
            
    def differentail_evolution(self):
        N = len(self.X)
        for i in range(N):
            a, b, c = np.random.choice(np.delete(np.arange(N), i), 3, replace=False)
            mutant = self.X[a] + self.F[i] * (self.X[b] - self.X[c])
            trial = np.copy(self.X[i])
            for j in range(self.dim):
                j_rand = np.random.randint(self.dim)
                if np.random.uniform() < self.CR[i] or j_rand == j:
                    trial[j] = mutant[j]
                    if trial[j] > self.xp[j]:
                        trial[j] = self.xp[j]
                    if trial[j] < self.xd[j]:
                        trial[j] = self.xd[j]
            yield trial
        # # 随机选取两个变量进行突变
        # k = np.random.randint(0,self.pop,2)
        # V = self.X + self.F * (self.X[k[0]]-self.X[k[1]])
        # # 超过范围的进行约束
        # for x in V:
        #     for p in range(len(self.X)):
        #         if(self.X[p] > self.xp[p]):
        #             self.X[p] = self.xp[p]
        #         if(self.X[p] < self.xd[p]):
        #             self.X[p] = self.xd[p]

        # # 交叉
        # is_cross = np.random.rand(self.pop * self.dim).reshape((self.pop,self.dim)) < self.P_CROSS
        # V = self.X * (1 - is_cross) + V * is_cross
        # return V
        
    def caculate_R(self,parent,offspring):
       
        if self.dominate(offspring, parent):
            return 1,0
        elif self.dominate(parent, offspring):
            return -1,1
        else:
            return 0,2
            
    def dominate(self,x,y):
        fx= self.base_obj_fun(x)
        fy= self.base_obj_fun(y)
        """判断解x是否支配解y"""
        return all([fx[i] <= fy[i] for i in range(len(fx))]) and any([fx[i] < fy[i] for i in range(len(fx))])
        
    def dominate_sort(self,XV):
        fitness = []
        self.pltx.clear()
        #print(XV)
        N = len(XV)
        #print(N)
        for i in range(N):
            fitness.append(self.obj_fun2(XV[i][0], XV[i][1]))
            
        fronts = self.non_dominated_sort(XV)
        #print(fronts)
        # 计算拥挤距离
        #i = 0
        for front in fronts:
            # if i == 0:
            #     print([XV[k] for k in front])
            if self.pltn - len(self.pltx) >= len(front):
                self.pltx.extend([XV[f] for f in front])
                continue
            assignments = np.array([fitness[i] for i in front])
            assignments = assignments.T
            assignments = assignments.tolist()
            distances = self.crowding_distance(assignments)
            sorted_indices = np.argsort(distances)
            
            for i in range(len(sorted_indices)):
                self.pltx.append(XV[sorted_indices[-i]])
                if self.pltn <= len(self.pltx):
                    break
            if self.pltn <= len(self.pltx):
                break
            #print(f"Crowding distances for front : {front}: {distances}")
        
        newX = np.array(self.pltx[:(self.elitesN)] + [self.pltx[i] for i in np.random.randint(
            self.elitesN,len(self.pltx),self.pop - self.elitesN)] )
        #print(newX,end='\n\n')
        return newX
        
    

    def dominates(self,x, y,epsilon=1e-6):
        """判断解x是否支配解y"""
        return all([x[i] <= y[i] for i in range(len(x))]) and any([x[i] - y[i] < -epsilon for i in range(len(x))])
    
    def non_dominated_sort(self,XV):
        fronts = [[]]  # 存储每一层的解
        N = len(XV)
        dominance_counts = [0] * N  # 记录每个解被多少个解支配
        dominated_solutions = [set() for _ in range(N)]  # 记录每个解支配的解
        
        # 计算支配关系
        for i in range(N):
            for j in range(N):
                if i!=j:
                    if self.dominate(XV[i],XV[j]):
                        dominated_solutions[i].add(j)
                    elif self.dominate(XV[j],XV[i]):
                        dominance_counts[i] += 1
        
        # 根据支配关系分配层级
        for i in range(N):
            if dominance_counts[i] == 0:
                fronts[0].append(i)
        
        
        #print([XV[x] for x in fronts[0]])
        
        # 动态分配剩余的解
        i = 0
        while len(fronts[i]) != 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    dominance_counts[q] -= 1
                    if dominance_counts[q] == 0:
                        next_front.append(q)
            fronts.append(next_front)
            i += 1
        
        return fronts[:-1]  # 去掉最后一个空的层级
    
    def crowding_distance(self,assignments):
        """
        计算拥挤距离
        :param assignments: 每个个体在各目标上的排序
        :return: 每个个体的拥挤距离
        """
        distances = [0.0] * len(assignments[0])  # 初始化每个个体的拥挤距离为0
        for rank in assignments:
            #print(rank)
            sorted_indices = np.argsort(rank)
            #print(sorted_indices)
            # 边缘个体（最大值和最小值）的拥挤距离设置为无穷大
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            f_max = rank[sorted_indices[-1]]
            f_min = rank[sorted_indices[0]]
            
            if f_max == f_min:
                continue
            
            # 计算每个个体与相邻个体的距离
            for i in range(1, len(rank)-1):
                distances[sorted_indices[i]] += (rank[sorted_indices[i+1]] - rank[sorted_indices[i-1]]) / (f_max - f_min)
        return distances
     
    
    
    
        