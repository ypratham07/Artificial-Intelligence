
#Function to Find Neighbours of Current State    
#The neighbours() function returns a list of possible neighbours 
def neighbours(node):
    #Finding the index of empty tile i.e Number 8
    empty_index=node.index(8)
        
    #Calculating the row and column 
    row=int(empty_index/3)
    col=int(empty_index%3)

    #Initializing moves to move left, right, up, down 
    moves=[(0,1),(0,-1),(1,0),(-1,0)]
    #declaring a list to store all the neighbours of current node i.e., current state
    valid_neighbors=[]

    #Iterating through each move i.e, left, right, up, down
    for move in moves:

        #calculating new row and new column for swap
        new_row=row + move[0]
        new_col=col + move[1]

        #In Case, new row and new column falls within the bounds of 3*3 matrix
        if (0<= new_row and new_row < 3) and (0<= new_col and new_col < 3):
            #Caculating new index form new row and new column 
            new_index= new_row * 3 + new_col
                
            #Taking a copy of current state i.e., current node
            new_node=node[:]
            #swapping the empty number i.e., 8 with the new position
            new_node[empty_index],new_node[new_index]=new_node[new_index],new_node[empty_index]
            #appending the new state to list of valid neighbours
            valid_neighbors.append(new_node)
        
    #returning the list of neighbours of current state, i.e. current node    
    return valid_neighbors

def astar(puzzle):
    #defining the goal state 
    goal_state=[0,1,2,3,4,5,6,7,8]
    
    #function to calculate heuristic value of a particular state using manhattan distance formula
    def heuristic(node):
        #Initializing total distance variable to zero
        total_distance=0
               
        #calculating manhattan distance of each tile 
        for i in range(0,9):
            #getting the index of each tile in current node
            a=node.index(i) 
            #getting index of each tile in goal node
            b=goal_state.index(i)

            #updating the total_distance of the state from the goal node
            total_distance=total_distance+ abs(int(a/3)-int(b/3)) + abs((a%3)-(b%3))

        #returning the heuristic cost i.e total distace of current node from the goal node  
        return total_distance
    
    #function to calculate the total cost to reach a particular state
    def f(node,g):
        #Total cost to reach a state is heuristic value of the state plus the cost required to reach the state form the initial starting state 
        return heuristic(node)+g
    
    #creating a list to maintain the tuples of all the explored states along with their heuristic values and path employed to reach the state
    open_set=[(heuristic(puzzle),puzzle,[puzzle])]
    #creating a set of elements to track the list of all the visited states
    closed_set=set()

    #iterating through the list of available states
    while open_set:
        #sorting the list of states so as to get the states with the minimum heuristic valeus first
        open_set.sort()
        #getting the state with minimum heuristic value
        current_tuple=open_set.pop(0)
        #since the current_tuple is a tuple containing 3 different values fetching each value of tuple in different variables
        heuristic_cost,current_node,path=current_tuple

        #In Case, when current_node i.e, the current state is equals to the goal state 
        if(current_node==goal_state):
            #declaring a new list to store the list of indexes with which the number 8 was swapped by to reach the goal state
            swap_index=[]

            #iterating through each element in path starting from the secon element as the first element is always the initial state
            for i in path[1:]:
                #appending the index of empty state i.e, number 8 in each state to the swap_index list
                swap_index.append(i.index(8))
            #return the list of indexes (swap_index) with which the empty tile (number 8) was swapped with in order to reach the goal state
            return swap_index
        
        #Otherwise, if the current node is not the goal state we are looking for then add the current state to the set of elements explored
        closed_set.add(tuple(current_node))

        #Iterating through each neighbour of current state 
        for neighbour in neighbours(current_node):
            #If the current neighbour in the iteration is not in set of explored states
            if tuple(neighbour) not in closed_set:
                #calculate the cost required to reach the neighbour state i.e. the current node in the iteration 
                g_value=len(neighbour)
                #calculate the f_value for the neighbour state
                f_value=f(neighbour,g_value)
                #add the current neighbour in the iteration to the open_set list 
                open_set.append((f_value,neighbour,path+[neighbour]))
    #In Case, all the states in open set are explored but none of them matches the goal state then no solution exists return empty list
    return []

a=astar([0, 4, 1, 3, 8, 2, 6, 7, 5])
print("A* Algorithm Output:",a)