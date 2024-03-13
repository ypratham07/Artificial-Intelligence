import math

v=[10,0,0,0,1] #store state values

# each item represents the action values for each state
q=[[10,10],[0,0],[0,0],[0,0],[1,1]] 

#Transition Function is 100% landing
p= 1.0
r= 5 # for both left/right actions
gamma=0.9

#Run Markov Decision process to iteratively update v and q based on 
#Bellman Backup Operator

for i in range(1000):
    v_new=v.copy()
    for j in range(1,len(v)-1):
        #Value for Q
        q[j][0]=gamma*v[j-1]+r 
        q[j][1]=gamma*v[j+1]+r

        #state value calcualtion
        v_new[j]=max(q[j][0],q[j][1])

    v=v_new.copy()
        


    print(v)




