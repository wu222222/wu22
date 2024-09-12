import numpy as np
import pandas as pd

class QLearningTable(object):
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,T = 0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.T = T
        self.actdic = {i:actions[i] for i in range(len(actions))}
        
        # print(self.actdic)
        
        self.q_table = pd.DataFrame(columns = self.actdic.keys(),dtype=np.float64)
        
    def choose_action(self,state):
        self.check_state_exist(state)
        # action selection
        
        
        # print(self.q_table.loc[state])
        
        actionkey= np.random.choice(self.q_table.loc[state].index,p=self.softmax(state))
        
        # if np.random.uniform() < self.epsilon:
        #     state_action = self.q_table.loc[state,:]
        #     actionkey = np.random.choice(state_action[state_action == np.max(state_action)].index)
        # else:
        #     actionkey = np.random.choice(list(self.actdic.keys()))
        
        return actionkey
    
    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.loc[s_,:].max()
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)
        
    def check_state_exist(self,state):
        if state not in self.q_table.index:
            #print(state,len(self.actions))
            # self.q_table = self.q_table.append(
            #     pd.Series(
            #         [0]*len(self.actions),
            #         index=self.q_table.columns,
            #         name=state,
            #         )
            #     )
            new_row = pd.Series(
                [0]*len(self.actions),
                index=self.q_table.columns,
                name=state,
                )
            #print(new_row)
            self.q_table = pd.concat([self.q_table,new_row.to_frame().T])
            #print(self.q_table)
            
    def softmax(self,state):
        e_softmax = np.exp(self.q_table.loc[state]/self.T)
        return e_softmax / np.sum(e_softmax)
        # print(self.epsilon)