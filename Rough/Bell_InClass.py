import numpy as np

v= np.zeros((4,4)) #state value initiall to 0 for non terminal states
q=np.zeros((4,16)) #initialize all the action values for each state is 0

#Interpret actions into code by considering how (x,y) index change
#"down","right","up", "left"
#action format: [y_change, x_change]
actions=[[0,1],[1,0],[0,-1],[-1,0]] #action ids

valid_locations=[[0,3], [1,0],[1,1],[1,2],[1,3], [2,0],[2,1],[2,2],[2,3], [3,0],[3,1],[3,2],[3,3]]

v[0,3]=100
v[3,0]=10
v[3,3]=-100

#define reward
r_0,r_1,r_2=0.5,-0.3,-0.3


for i in range(100):
    v_new=v.copy() #make a copy of last iteration's state values
    for y in range(1,3):
        for x in range(4):
            for a in range(len(actions)):
                #a is the ideal landing direction with 80% probability
                
                #a_1 is one neighbour with 10% probability 
                a_1= a - 1
                
                if a_1 <0:
                    a_1=3
                #a_2 is another neighbour action with 10% probability
                a_2= a + 1

                if (a_2<3):
                    a_2=0

                #Find out the possible three landing state
                n_loc_0= (np.array([y,x]) + np.array(actions[a])).tolist()
                n_loc_1= (np.array([y,x]) + np.array(actions[a_1])).tolist()
                n_loc_2= (np.array([y,x]) + np.array(actions[a_2])).tolist()

                #double check these new locations
                if(n_loc_0 not in valid_locations):
                    n_loc_0=[y,x]
                    r_0=-0.3
                if(n_loc_1 not in valid_locations):
                    n_loc_1=[y,x]
                if(n_loc_2 not in valid_locations):
                    n_loc_2=[y,x]
                
                v_new[y,x]=gama*v[n_loc_0]*0.8 + gama *v[n_loc_1]*0.1


print(v)
