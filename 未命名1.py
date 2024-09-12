from q_brain import QLearningTable
from Envm import Env

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(15)

RAN = 20
POP = 50
#ELITES = 
ACTIONS = [[-0.1,0.1],[0.1,0.1],[0,0]]


def RLMODE():
    q_table = QLearningTable(actions=ACTIONS)
    #print(q_table.q_table)
    env = Env(ACTIONS,popular_size=POP)
    #print(env.obj_fun([1,1]))
    for episode in range(RAN):
        S = np.random.randint(0,high=3, size=env.pop)
        new_X = []
        for i,x in enumerate(env.X):
            Akey = q_table.choose_action(S[i])
            
            #print(A)
            #print(env.F)
            
            env.get_env_feedback(i,q_table.actdic[Akey])
            v = next(env.differentail_evolution())
            
            #print(x,v)
            
            R,S_ = env.caculate_R(x, v)
            
            #if R == 1:
            v = v
            env.F[i] = np.random.uniform()
            env.CR[i] = np.random.uniform()
                #S[i] = np.random.randint(0,high=3)
            # else:
            #     v = x
            
            q_table.learn(S[i], Akey, R, S_)
            S[i] = S_
            
            new_X.append(v.tolist())
        #env.X = new_X
        #print((new_X+env.X.tolist())[0])
        env.X = env.dominate_sort(new_X+env.X.tolist())
        
        
    df = pd.DataFrame([env.base_obj_fun(x) for x in env.pltx [:env.pop] ])
    #print(df)
    #print(env.X[:30])
    plt.scatter(df[0], df[1])
    plt.show()
    return q_table

if __name__ == "__main__":
    
    #print(act)
    q_table = RLMODE()
    print('\r\nQ-table:\n')
    print(q_table.q_table)
    
    
    