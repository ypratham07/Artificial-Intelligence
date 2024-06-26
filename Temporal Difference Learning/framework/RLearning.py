
"""
@author: Pratham Yadav
@email: yadavp9@udayton.edu
@date: 03-31-24
"""
import random
import numpy as np
import math as mth

# The state class
class State:
    def __init__(self, angle1=0, angle2=0):
        self.angle1 = angle1
        self.angle2 = angle2

class ReinforceLearning:

    #
    def __init__(self, unit=5):

        ####################################  Needed: here are the variable to use  ################################################

        # The crawler agent
        self.crawler = 0

        # Number of iterations for learning
        self.steps = 1000

        # learning rate alpha
        self.alpha = 0.2

        # Discounting factor
        self.gamma = 0.95

        # E-greedy probability
        self.epsilon = 0.1

        self.Qvalue = []  # Update Q values here
        self.unit = unit  # 5-degrees
        self.angle1_range = [-35, 55]  # specify the range of "angle1"
        self.angle2_range = [0, 180]  # specify the range of "angle2"
        self.rows = int(1 + (self.angle1_range[1] - self.angle1_range[0]) / unit)  # the number of possible angle 1
        self.cols = int(1 + (self.angle2_range[1] - self.angle2_range[0]) / unit)  # the number of possible angle 2

        ########################################################  End of Needed  ################################################



        self.pi = [] # store policies
        self.actions = [-1, +1] # possible actions for each angle

        # Controlling Process
        self.learned = False



        # Initialize all the Q-values
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.Qvalue.append(row)



        # Initialize all the action combinations
        self.actions = ((-1, -1), (-1, 0), (0, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


        # Initialize the policy
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(random.randint(0, 8))
            self.pi.append(row)





    # Reset the learner to empty
    def reset(self):
        self.Qvalue = [] # store Q values
        self.R = [] # not working
        self.pi = [] # store policies

        # Initialize all the Q-values
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.Qvalue.append(row)

        # Initiliaize all the Reward
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.R.append(row)

        # Initialize all the action combinations
        self.actions = ((-1, -1), (-1, 0), (0, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


        # Initialize the policy
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(random.randint(0, 8))
            self.pi.append(row)

        # Controlling Process
        self.learned = False

    # Set the basic info about the robot
    def setBot(self, crawler):
        self.crawler = crawler


    def storeCurrentStatus(self):
        self.old_location = self.crawler.location
        self.old_angle1 = self.crawler.angle1
        self.old_angle2 = self.crawler.angle2
        self.old_contact = self.crawler.contact
        self.old_contact_pt = self.crawler.contact_pt
        self.old_location = self.crawler.location
        self.old_p1 = self.crawler.p1
        self.old_p2 = self.crawler.p2
        self.old_p3 = self.crawler.p3
        self.old_p4 = self.crawler.p4
        self.old_p5 = self.crawler.p5
        self.old_p6 = self.crawler.p6

    def reverseStatus(self):
        self.crawler.location = self.old_location
        self.crawler.angle1 = self.old_angle1
        self.crawler.angle2 = self.old_angle2
        self.crawler.contact = self.old_contact
        self.crawler.contact_pt = self.old_contact_pt
        self.crawler.location = self.old_location
        self.crawler.p1 = self.old_p1
        self.crawler.p2 = self.old_p2
        self.crawler.p3 = self.old_p3
        self.crawler.p4 = self.old_p4
        self.crawler.p5 = self.old_p5
        self.crawler.p6 = self.old_p6



    def updatePolicy(self):
        # After convergence, generate policy y
        for r in range(self.rows):
            for c in range(self.cols):
                max_idx = 0
                max_value = -1000
                for i in range(9):
                    if self.Qvalue[r][9 * c + i] >= max_value:
                        max_value = self.Qvalue[r][9 * c + i]
                        max_idx = i

                # Assign the best action
                self.pi[r][c] = max_idx


    # This function will do additional saving of current states than Q-learning
    def onLearningProxy(self, option):
        self.storeCurrentStatus()
        if option == 0:
            self.onMonteCarlo()
        elif option == 1:
            self.onTDLearning()
        elif option == 2:
            self.onQLearning()
        self.reverseStatus()


        # Turn off learned
        self.learned = True



    # For the play_btn uses: choose an action based on the policy pi
    def onPlay(self, ang1, ang2, mode=1):

        epsilon = self.epsilon

        ang1_cur = ang1
        ang2_cur = ang2

        # get the state index
        r = int((ang1_cur - self.angle1_range[0]) / self.unit)
        c = int((ang2_cur - self.angle2_range[0]) / self.unit)

        # Choose an action and udpate the angles
        idx, angle1_update, angle2_update = self.chooseAction(r=r, c=c)
        ang1_cur += self.unit * angle1_update
        ang2_cur += self.unit * angle2_update

        return ang1_cur, ang2_cur



    ####################################  Needed: here are the functions you need to use  ################################################


    # This function is similar to the "runReward()" function but without returning a reward.
    # It only update the robot position with the new input "angle1" and "angle2"
    def setBotAngles(self, ang1, ang2):
        self.crawler.angle1 = ang1
        self.crawler.angle2 = ang2
        self.crawler.posConfig()



    # Given the current state, return an action index and angle1_update, angle2_update
    # Return valuse
    #  - index: any number from 0 to 8, which indicates the next action to take, according to the e-greedy algorithm
    #  - angle1_update: return the angle1 new value according to the action index, one of -1, 0, +1
    #  - angle2_update: the same as angle1

    def chooseAction(self, r, c):
        # Implement the epsilon-greedy policy
        if random.random() < self.epsilon:
            
            # Explore: choose a random action
            idx = random.randint(0, 8)
        
        else:
            
            # Exploit: choose the action with the highest Q-value
            max_q = float('-inf')
            max_idx = 0
            for i in range(9):
                q_value = self.Qvalue[r][9 * c + i]
                if q_value > max_q:
                    max_q = q_value
                    max_idx = i
            idx = max_idx

        # Compute the angle updates based on the chosen action
        angle1_update = self.actions[idx][0]
        angle2_update = self.actions[idx][1]

        # Handle the case where the updated angles go beyond the specified range
        if angle1_update * self.unit + self.crawler.angle1 < self.angle1_range[0] or angle1_update * self.unit + self.crawler.angle1 > self.angle1_range[1]:
            angle1_update = 0
        if angle2_update * self.unit + self.crawler.angle2 < self.angle2_range[0] or angle2_update * self.unit + self.crawler.angle2 > self.angle2_range[1]:
            angle2_update = 0

        return idx, angle1_update, angle2_update


    # Method 1: Monte Carlo algorithm
    def onMonteCarlo(self):
        # You don't need to implement this function
        return



    # Method 2: Temporal Difference based on SARSA
    def onTDLearning(self):
        for _ in range(self.steps):
            
            trajectory = []
            state = State(self.crawler.angle1, self.crawler.angle2)
            done = False
            count = 0
            while count<=self.steps:
                
                # Choose an action using the epsilon-greedy policy
                r = int((state.angle1 - self.angle1_range[0]) / self.unit)
                c = int((state.angle2 - self.angle2_range[0]) / self.unit)
                idx, angle1_update, angle2_update = self.chooseAction(r, c)
                trajectory.append((state, idx))

                # Update the agent's angles and check if the episode is done
                new_angle1 = state.angle1 + angle1_update * self.unit
                new_angle2 = state.angle2 + angle2_update * self.unit
                self.setBotAngles(new_angle1, new_angle2)
                state = State(new_angle1, new_angle2)
                count = count + 1
                if self.crawler.location[0] <= 0:
                    done = True

            # Update the Q-values along the trajectory using the SARSA update rule
            G = self.crawler.location[0]  # Compute the return
            for i in range(len(trajectory) - 1, -1, -1):
                state, action_idx = trajectory[i]
                r = int((state.angle1 - self.angle1_range[0]) / self.unit)
                c = int((state.angle2 - self.angle2_range[0]) / self.unit)
                self.Qvalue[r][9 * c + action_idx] += self.alpha * (G - self.Qvalue[r][9 * c + action_idx])
                G = self.gamma * G
            
            #Printing the G Values in order to debug the output
            print(G)
            print(count)
        
        # Update the policy based on the learned Q-values
        self.updatePolicy()



    # Method 3: Bellman operator based Q-learning
    def onQLearning(self):
        # You don't have to work on it for the moment
        return




