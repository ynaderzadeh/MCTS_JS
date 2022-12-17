#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:54:20 2022

@author: yasnader
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:41:21 2022

@author: yasnader
"""

import numpy as np
from itertools import product
from numpy.typing import NDArray
import copy
from numpy import linalg as LA

np.random.seed(4)

NUM_MCHN = 15
HP =  3
BUDGET = 5000

# jobs_data = [  # task = (machine_id, processing_time).
#     [(0, 3), (1, 2), (2, 2)],  # Job0
#     [(0, 2), (2, 1), (1, 4)],  # Job1
#     [(1, 4), (2, 3)]  # Job2
# ]




# jobs_data =[
# [(2,44),(3, 5),(5, 58),(4,97),(0,9),(7,84),(8,77),(9,96),(1,58),(6,89)],
# [(4,15),(7,31),(1,87),(8,57),(0, 77),(3,85),(2 ,81),(5 ,39),(9,73),(6,21)],
# [(9,82),(6,22),(4,10),(3,70),(1,49),(0,40),(8,34),(2,48),(7,80),(5,71)],
# [(1,91),(2,17),(7,62),(5,75),(8,47),(4,11),(3,7),(6,72),(9,35),(0,55)],
# [(6,71),(1,90),(3,75),(0,64),(2,94),(8,15),(4,12),(7,67),(9,20),(5,50)],
# [(7,70),(5,93),(8,77),(2,29),(4,58),(6,93),(3,68),(1,57),(9,7),(0,52)],
# [(6,87),(1,63),(4,26),(5,6),(2,82),(3,27),(7,56),(8,48),(9,36),(0,95)],
# [(0,36),(5,15),(8,41),(9,78),(3,76),(6,84),(4,30),(7,76),(2,36),(1,8)],
# [(5,88),(2,81),(3,13),(6,82),(4,54),(7,13),(8,29),(9,40),(1,78),(0,75)],
# [(9,88),(4,54),(6,64),(7,32),(0,52),(2,6),(8,54),(5,82),(3,6),(1,26)]

# ]



la21 = "2	34	3	55	5	95	9	16	4	21	6	71	0	53	8	52	1	21	7	26 \
3	39	2	31	0	12	1	42	9	79	8	77	6	77	5	98	4	55	7	66 \
1	19	0	83	3	34	4	92	6	54	9	79	8	62	5	37	2	64	7	43 \
4	60	2	87	8	24	5	77	3	69	7	38	1	87	6	41	9	83	0	93 \
8	79	9	77	2	98	4	96	3	17	0	44	7	43	6	75	1	49	5	25 \
8	35	7	95	6	9	9	10	2	35	1	7	5	28	4	61	0	95	3	76 \
4	28	5	59	3	16	9	43	0	46	8	50	6	52	7	27	2	59	1	91 \
5	9	4	20	2	39	6	54	1	45	7	71	0	87	3	41	9	43	8	14 \
1	28	5	33	0	78	3	26	2	37	7	8	8	66	6	89	9	42	4	33 \
2	94	5	84	6	78	9	81	1	74	3	27	8	69	0	69	7	45	4	96 \
1	31	4	24	0	20	2	17	9	25	8	81	5	76	3	87	7	32	6	18 \
5	28	9	97	0	58	4	45	6	76	3	99	2	23	1	72	8	90	7	86 \
5	27	9	48	8	27	7	62	4	98	6	67	3	48	0	42	1	46	2	17 \
1	12	8	50	0	80	2	50	9	80	3	19	5	28	6	63	4	94	7	98 \
4	61	3	55	6	37	5	14	2	50	8	79	1	41	9	72	7	18	0	75"

# la21 = "0	12	2	94	3	92	4	91	1	7 \
# 1	19	3	11	4	66	2	21	0	87 \
# 1	14	0	75	3	13	4	16	2	20 \
# 2	95	4	66	0	7	3	7	1	77 \
# 1	45	3	6	4	89	0	15	2	34 \
# 3	77	2	20	0	76	4	88	1	53 \
# 2	74	1	88	0	52	3	27	4	9 \
# 1	88	3	69	0	62	4	98	2	52 \
# 2	61	4	9	0	62	1	52	3	90 \
# 2	54	4	5	3	59	1	15	0	88"



la21 = la21.split()

jobs_data = []

la21 = [int(_) for _ in la21]

import numpy as np

la21 = np.array(la21).reshape((15,20))

la21 = list(la21)

for j in la21:
    temp = []
    for t in range(0,20,2):
        
        temp.append(tuple(j[t:t+2]))
    jobs_data.append(temp)




def initial_stck(jobs_data):
    
    stack = []
    
    for j in jobs_data:
        
        stack.append((j[0][0],j[0][1],len(j)-1,0))
    return stack
        
def upper_bound(jobs_data):
    
    ub = 0
    
    for j in jobs_data:
        
        for t in j:
            
            ub += t[1]
    return ub
    

initial_stack = initial_stck(jobs_data)
initila_mchn_state = [[0] for _ in range(NUM_MCHN)]
upb = upper_bound(jobs_data)


class State_Node:
    
    def __init__(self, mchn_state, stck_state, parent, time):
        
        
        #self.exp_p = exp_p
        self.acts = []
        self.mchn_state = mchn_state
        self.stck_state = stck_state
        
        self.num_visit = 0
        self.parent = parent
        self.time = time #max(time,min(self.stck_state, key = lambda x : x[-1])[-1])
        self.val = 0
        
    #@staticmethod
    def set_time(stack):  
        pass
        
    
    def action_maker(self):
        
        
        actions = []
        
        for m in enumerate(self.mchn_state):
            
            if m[1][0] > self.time:
                actions.append(['null'])
                
            else:
                #print('machine: {}'.format(m[0]))
                m_temp = ['null']
                
                for s_idx, s in enumerate(self.stck_state):
                    
                    #print('job: {}'.format(s_idx))
                    
                    if s != -1:
                    
                        if s[0] == m[0] and self.time >= s[-1]:
                            m_temp.append(str(s_idx)+"_"+str(s[2])+"_"+str(s[1]))
                        
                # if m_temp == []:
                #     m_temp.append('null')
                        
                actions.append(m_temp)
            
        
        self.acts = [Action_Node(act_state, self, self.time) for act_state in list(product(*actions))][1:]
        
            
    def terminal(self):
        
        for s in self.stck_state:
            
            if s != -1:
                return False
        return True
        
    def action_selection(self):
        
           
        i = np.argmax(np.array([a.cal_ucb() for a in self.acts]))
        
        #self.acts[i].parent.num_visit += 1
        self.acts[i].visited = True
        
        return self.acts[i]
    
    def random_action_section(self):
        
        # print(len(self.acts))
        # print(self.stck_state)
        
        i = np.random.choice(len(self.acts), 1)[0]
        
        #self.acts[i].num_chosen += 1
        
        return self.acts[i]
    
    def stack_job(self):
        
        self.top_stack = [[] for _ in range(NUM_MCHN)]
        for j in enumerate(self.stck_state):
            
            if j[1] != -1:
                self.top_stack[j[1][0]].append(j[0])
                
    
            
            
    
class Action_Node:
    
    def __init__(self, a_state, parent, time):
        
        
        #self.exp_p = exp_p
        self.a_state = a_state
        self.ucb = np.inf
        self.parent = parent
        self.time = time
        self.visited = False
        self.child_state = None
        
    @staticmethod    
    def stack_pop(job, pri, jobs_data, time):
        
        if pri == 0:
            
            return -1
        else:
            
            return (jobs_data[job][-pri][0], jobs_data[job][-pri][1],pri-1, 
                    time+jobs_data[job][-pri-1][1] )
        
        
    @staticmethod 
    def UCB1(N, ni, v, c = np.sqrt(HP)):
        
        if ni == 0:
            
            return np.Inf
        else:
        
            return (v/ni) + c * np.sqrt(np.log(N)/(ni))
        
    def find_min_time(self, stack):
        
        
        min_time = upb
        
        for j in stack:
            if j != -1:
                if j[3] < min_time:
                    #print(j[3])
                    min_time = j[3]
        if min_time == upb:
            return self.time
        else:
            return min_time
        
    @staticmethod     
    def update_stack(stack, n_state):
        
        for j_x, j in enumerate(stack):
            
            if j != -1:
                w_time = max(n_state[j[0]][0], j[3])
                
                stack[j_x] = (stack[j_x][0],stack[j_x][1],stack[j_x][2],w_time)
                
        return stack
        

    def act_state(self):
        
        
        n_state = []
        stack = copy.deepcopy(self.parent.stck_state)
        
        # min_time = upb
        
        self.parent.stack_job()
        
        for a in enumerate(self.a_state):
            
            if a[1] != "null":
                
                job = int(a[1].split("_")[0])
                pri = int(a[1].split("_")[1])
                
                stack[job] = self.stack_pop(job, pri, jobs_data, self.time)
                # if stack[job] != -1:
                #     if stack[job][3] < min_time:
                #         min_time = stack[job][3]
                # else:
                #     min_time  = self.time + int(a[1].split("_")[2])
                #tim_fwd = max(self.parent.mchn_state[a[0]][0],self.time) + float(a[1].split("_")[-1])
                for j in self.parent.top_stack[a[0]]:
                    if j != job:
                        if stack[job] != -1:
                            if stack[job][3] > stack[j][3]:
                            
                                stack[j] =  (stack[j][0],stack[j][1],stack[j][2],stack[job][3])
                        else:
                            t_wait = max(stack[j][3], self.time + int(a[1].split("_")[2]))
                            stack[j] =  (stack[j][0],stack[j][1],stack[j][2],t_wait)
                    
                tim_fwd = self.time + int(a[1].split("_")[-1])
            else:
                
                tim_fwd = self.parent.mchn_state[a[0]][0]
                
            n_state.append([tim_fwd])
        
        
        stack = self.update_stack(stack,n_state)
        min_time = self.find_min_time(stack)
        if min_time <= self.time:
            min_time = self.time+1
        
        #self.child_state = State_Node(n_state, stack, self, self.time+1)
        self.child_state = State_Node(n_state, stack, self, min_time)
        
    def cal_ucb(self):
        
        
        
        if self.visited:
        
            self.ucb = self.UCB1(self.parent.num_visit,self.child_state.num_visit,
                            self.child_state.val)
        return self.ucb 




def reward(upb,state):
    
    
    max_ms = 0
    
    for m in state.mchn_state:
        
        if m[0] > max_ms:
            
            max_ms = m[0]
    return 10*(upb - max_ms)/upb, upb - max_ms #upb - max_ms, upb - max_ms# (upb - max_ms)/upb, upb - max_ms



def rollout(act):
    
    
    if act.child_state == None:
        act.act_state()
    
    while not act.child_state.terminal():
        
        if act.child_state.acts == []:
            act.child_state.action_maker()
        
        act = act.child_state.random_action_section()
        
        if act.child_state == None:
            act.act_state()
    r = reward(upb, act.child_state)
    
    return r, act.child_state



def back_propagate(state_r, r):
    
    temp_state = state_r
    
    
    
    while temp_state.parent:
        
        temp_state.num_visit += 1
        temp_state.val += r
        
        
        
        temp_state = temp_state.parent.parent

        
    temp_state.num_visit += 1
    temp_state.val += r 
        

def MCTS(budget, root) :
    
    ms = upb
    ms_ = upb
    root.action_maker()
    #print("length roo acts {}".format(len(root.acts)))
    for _ in range(budget):
        
        if not _ % 250:
            print(_)
        
        act = root.action_selection()
        
        #print(act.a_state)
        flag = True
        while act.visited:
            
            
            #print("inside while {}".format(_))
            
            if act.child_state != None:
                
                state = act.child_state
                if state.acts == []:
                    state.action_maker()
            else:
                act.act_state()
                state = act.child_state
                if state.acts == []:
                    state.action_maker()
            if state.terminal():
                r = reward(upb,state)
                back_propagate(state, r[0])
                if ms_ - r[1] < ms:
                    ms = ms_ - r[1]
                    state_min = state
                flag = False
                break
                
                
            if state.num_visit > 0:
                
                act = state.action_selection()
            else:
                # print(state.stck_state)
                # print(state.parent.a_state)
                # print(state.time)
                # print(state.mchn_state)
                act = state.random_action_section()
        if flag:    
            r, state = rollout(act)
            back_propagate(act.parent, r[0])
            
            if ms_ - r[1] < ms:
                ms = ms_ - r[1]
                state_min = state
        # print(_)
        # print(ms_ - r)#, state.mchn_state)
    print(ms)
            
    return state_min
    
    
    
root  = State_Node(initila_mchn_state, initial_stack, None, 0)    
# root.action_maker()
# root.acts[0].act_state()
# root.action_selection().a_state

s = MCTS(BUDGET, root)






def print_root(root):
    
    state = root
    while not state.terminal():
        
        print(state.time)
        print(state.mchn_state)
        print(state.stck_state)
        #i = np.argmax(np.array([a.ucb for a in state.acts]))
        i = np.argmax(np.array([a.child_state.num_visit for a in state.acts]))
        print(state.acts[i].a_state)
        
        state = state.acts[i].child_state
    print(state.time)
    print(state.mchn_state)
    print(state.stck_state)
   

#print_root(root)




def bottom_up(state):
    print(state.time)
    while state.parent:
        
        print(state.mchn_state)
        print(state.stck_state)
        print(state.parent.time)
        print(state.parent.a_state)
        state = state.parent.parent
    #print(state.time)    
    print(state.mchn_state)
    #print(state.stck_state)
    #print(state.a_state) 
x = [a.child_state.num_visit for a in root.acts]   
#bottom_up(s)


def save_gant_dic(state):
    
    pass




def update_stack(stack, n_state):
    
    for j_x, j in enumerate(stack):
        
        if j != -1:
            w_time = max(n_state[jobs_data[j[0]][-j[2]-1][0]][0], j[3])
            
            stack[j_x] = (stack[j_x][0],stack[j_x][1],stack[j_x][2],w_time)
            
    return stack
    



#x = [a.child_state.num_visit for a in root.acts[376].child_state.acts]
#566
# Solution:
# Optimal Schedule Length: 842.0
# Machine 0: job_7_task_0   job_0_task_4   job_9_task_4   job_1_task_4   job_4_task_3   job_2_task_5   job_3_task_9   job_8_task_9   job_6_task_9   job_5_task_9   
#            [0,36]         [259,268]      [268,320]      [320,397]      [397,461]      [492,532]      [559,614]      [614,689]      [690,785]      [790,842]      
# Machine 1: job_3_task_0   job_1_task_2   job_4_task_1   job_6_task_1   job_2_task_4   job_8_task_8   job_0_task_8   job_5_task_7   job_7_task_9   job_9_task_9   
#            [0,91]         [91,178]       [178,268]      [268,331]      [443,492]      [524,602]      [663,721]      [721,778]      [778,786]      [809,835]      
# Machine 2: job_0_task_0   job_3_task_1   job_8_task_1   job_9_task_5   job_5_task_3   job_6_task_4   job_4_task_4   job_1_task_6   job_2_task_7   job_7_task_8   
#            [0,44]         [91,108]       [108,189]      [320,326]      [331,360]      [363,445]      [465,559]      [562,643]      [643,691]      [691,727]      
# Machine 3: job_0_task_1   job_8_task_2   job_7_task_4   job_4_task_2   job_2_task_3   job_3_task_6   job_6_task_5   job_1_task_5   job_5_task_6   job_9_task_8   
#            [44,49]        [189,202]      [222,298]      [298,373]      [373,443]      [443,450]      [450,477]      [477,562]      [616,684]      [803,809]      
# Machine 4: job_1_task_0   job_9_task_1   job_0_task_3   job_6_task_2   job_2_task_2   job_8_task_4   job_3_task_5   job_5_task_4   job_7_task_6   job_4_task_6   
#            [0,15]         [89,143]       [162,259]      [331,357]      [357,367]      [367,421]      [421,432]      [448,506]      [506,536]      [604,616]      
# Machine 5: job_8_task_0   job_7_task_1   job_0_task_2   job_5_task_1   job_3_task_3   job_6_task_3   job_9_task_7   job_1_task_7   job_4_task_9   job_2_task_9   
#            [0,88]         [88,103]       [103,161]      [161,254]      [254,329]      [357,363]      [432,514]      [643,682]      [710,760]      [771,842]      
# Machine 6: job_4_task_0   job_6_task_0   job_9_task_2   job_8_task_3   job_2_task_1   job_7_task_5   job_3_task_7   job_5_task_5   job_0_task_9   job_1_task_9   
#            [0,71]         [73,160]       [160,224]      [246,328]      [328,350]      [350,434]      [450,522]      [522,615]      [721,810]      [810,831]      
# Machine 7: job_1_task_1   job_5_task_0   job_3_task_2   job_9_task_3   job_0_task_5   job_8_task_5   job_6_task_6   job_7_task_7   job_4_task_7   job_2_task_8   
#            [15,46]        [46,116]       [116,178]      [224,256]      [289,373]      [431,444]      [477,533]      [536,612]      [616,683]      [691,771]      
# Machine 8: job_7_task_2   job_1_task_3   job_5_task_2   job_3_task_4   job_9_task_6   job_8_task_6   job_0_task_6   job_2_task_6   job_4_task_5   job_6_task_7   
#            [103,144]      [179,236]      [254,331]      [331,378]      [378,432]      [449,478]      [478,555]      [555,589]      [589,604]      [606,654]      
# Machine 9: job_9_task_0   job_7_task_3   job_2_task_0   job_8_task_7   job_3_task_8   job_0_task_7   job_6_task_8   job_4_task_8   job_1_task_8   job_5_task_8   
#            [0,88]         [144,222]      [222,304]      [480,520]      [522,557]      [557,653]      [654,690]      [690,710]      [710,783]      [783,790]      