import numpy as np

POPULATION = 10  # 种群数
F = 0.7 # 差分缩放因子
P_CROSS = 0.9 # 交叉概率
MAX_ITER = 50 # 最大迭代次数
lb = np.array([-10]) # 下限
ub = np.array([10]) # 上限

# 目标函数
def obj_fun(u):
    f = np.cos(u) * np.exp(-(u - np.pi) ** 2)
    return f[0]

# 初始化种群
s = np.zeros((POPULATION,len(lb)))
fitness = np.zeros(POPULATION)
for i in range(POPULATION):
    s[i] = lb + (ub - lb) * np.random.rand(len(lb))
    fitness[i] = obj_fun(s[i])

i = 0
f_opt = np.min(fitness)
u_opt = s[np.argmin(fitness)]

while i < MAX_ITER:
    i += 1
    # 随机选取两个变量进行突变
    k = np.random.randint(0,POPULATION,2)
    v = s + F * (s[k[0]]-s[k[1]])
    # 超过范围的进行约束
    for x in v:
        for p in range(len(x)):
            if(x[p] > ub[p]):
                x[p] = ub[p]
            if(x[p] < lb[p]):
                x[p] = lb[p]

    # 交叉
    is_cross = np.random.rand(POPULATION * len(lb)).reshape((POPULATION,len(lb))) < P_CROSS
    v = s * (1 - is_cross) + v * is_cross
    # 新生的的比原来的好就进行替换
    for p in range(POPULATION):
        f_new  = obj_fun(v[p])
        if(f_new <= fitness[p]):
            s[p] = v[p]
            fitness[p] = f_new
        if(f_new <= f_opt):
            f_opt = f_new
            u_opt = v[p]
            print(u_opt,f_opt,sep='\t')

print(u_opt)
print(f_opt)
