import random
from matplotlib.pylab import rand
import numpy as np

# 初始化参数
np.random.seed(42)
num_states = 3     # 状态数量
num_actions = 3    # 动作数量
discount_factor = 0.9  # 折扣因子
T = 0.9            # Softmax温度参数

# 初始化Q表，F，CR
Q_table = np.zeros((num_states, num_actions))

# SoftMax函数
def softmax(Q_values, T):
    exp_values = np.exp(Q_values / T)
    return exp_values / np.sum(exp_values)

# 选择动作
def choose_action(state):
    probabilities = softmax(Q_table[state], T)
    return np.random.choice(num_actions, p=probabilities)

# 更新Q表
def update_q_table(current_state, next_state, discount_factor, action, reward):
    learning_rate = 0.1
    Q_table[current_state, action] += learning_rate * (reward 
                                                       + discount_factor * np.max(Q_table[next_state]) 
                                                       - Q_table[current_state, action])

# 差分进化算法的变异和交叉
def differential_evolution(F, CR, X):
    N = len(X)
    for i in range(N):
        a, b, c = np.random.choice(np.delete(np.arange(N), i), 3, replace=False)
        mutant = X[a] + F[i] * (X[b] - X[c])
        trial = np.copy(X[i])
        for j in range(len(X[i])):
            j_rand = np.random.randint(len(X[i]))
            if random.random() < CR[i] or j_rand == j:
                trial[j] = mutant[j]
        yield trial

# 奖励机制
def calculate_reward(parent, offspring):
    if objective_function(offspring) < objective_function(parent):
        return 1
    elif objective_function(offspring) > objective_function(parent):
        return -1
    else:
        return 0

# 伪目标函数
def objective_function(vector):
    return np.sum(vector ** 2)

# 初始种群
population_size = 100
dim = 10
X = np.random.rand(population_size, dim)  # 初始种群
F = np.random.rand(population_size)
CR = np.random.rand(population_size)
current_state = np.random.randint(3, size=population_size)  # 初始状态

# 迭代次数
iterations = 100
new_results = []

# 开始迭代
for t in range(iterations):
    new_population = []
    for i, x in enumerate(X):
        action = choose_action(current_state[i])  # 根据状态选择动作
        
        # 根据动作更新F和CR
        F[i] += [-0.1, 0.1, 0][action]
        CR[i] += [0.1, 0.1, 0][action]
        
        if F[i] < 0 or F[i] > 1:
            F[i] = random.random()
        if CR[i] < 0 or CR[i] > 1:
            CR[i] = random.random()
        
        # 生成后代
        trial_vector = next(differential_evolution(F, CR, X))
        
        # 计算奖励并更新Q表
        reward = calculate_reward(x, trial_vector)
        
        # 简化的状态转移
        next_state = [2, 0, 1][reward]
        
        # 如果后代优于父代，则后代成为新的状态
        trial_vector = trial_vector if reward == 1 else x
        
        # 更新Q表以及状态
        update_q_table(current_state[i], next_state, discount_factor, action, reward)
        current_state[i] = next_state
        
        # 保存后代到新种群
        new_population.append(trial_vector)
    
    new_results.append(objective_function(trial_vector))
    
    # 更新种群
    X = np.array(new_population)

# 输出结果
print(new_results)

import matplotlib.pyplot as plt

plt.plot(new_results)
plt.xlabel('Iteration')
plt.ylabel('Objective Function Value')
plt.title('Convergence Plot')
plt.show()
