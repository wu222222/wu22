from q_brain import QLearningTable
from Envm import Env

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

np.random.seed(15)

RAN = 35
POP = 500
xN = 5 #自变量维度数
#ELITES = 
ACTIONS = [[-0.1,0.1],[0.1,0.1],[0,0]]


def RLMODE():
    q_table = QLearningTable(actions=ACTIONS)
    #print(q_table.q_table)
    env = Env(ACTIONS,popular_size=POP,xN=xN)
    #print(env.obj_fun([1,1]))
    for episode in range(RAN):
        epi_s = time.time()
        epi = 1
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
            
            
            v = v
            if R == 1:
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
        
        
        epi_e = time.time()
        df = pd.DataFrame([env.base_obj_fun(x) for x in env.pltx [:env.pop] ])
        plt.scatter(df[0], df[1])
        #plt.xticks(np.arange(0,1,0.2))
        #plt.yticks(np.arange(0.25,1.25,0.25))
        plt.title(f"ran:{episode} PerCostTime:{round(epi_e - epi_s,4)} MayCostTime:{round((RAN-epi)*(epi_e - epi_s),4)}")
        plt.show()
        #time.sleep(0.2)
        epi+=1
        
    df = pd.DataFrame([env.base_obj_fun(x) for x in env.pltx [:env.pop] ])
    #print(df)
    #print(env.X[:30])
    plt.scatter(df[0], df[1])
    
    ed_time = time.time()
    
    plt.title(f"xN:{xN} RAN:{RAN} POP:{POP} CostTime:{round(ed_time - st_time,4)}")
    plt.show()
    
    df.to_csv("D:\\py\\test.csv",header=False,index=False)
    #print(env.func.constraints_func([0.9001995503509116,0.5589939250456916]))
    return q_table

if __name__ == "__main__":
    st_time = time.time()
    #print(act)
    q_table = RLMODE()

    print('\r\nQ-table:\n')
    print(q_table.q_table)
    #print("Cost Time:{}".format(round(ed_time - st_time,4) ))
    
    