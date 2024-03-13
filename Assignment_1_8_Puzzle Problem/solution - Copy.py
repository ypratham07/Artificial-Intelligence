#This is the only file you need to work on. You do NOT need to modify other files

# Below are the functions you need to implement. For the first project, you only need to finish implementing iddfs() 
# ie iterative deepening depth first search


# here you need to implement the Iterative Deepening Search Method
def iterativeDeepening(puzzle):
    
    #Depth First Search
    def dfs(current_state,depth, path, goal_state):
        
        #When depth=0 and durrent state is goal state return the sequence of swaps 
        if(depth==0 and current_state==goal_state):
            #declaring a list to store all the swap sequences
            swap_list=[]

            #iterating through each state that led to goal state 
            for i in path[1:]:
                #finding the index of number 8 i.e number which represents blank space
                swap_list.append(i.index(8))
            #returning the list of swap sequences
            return swap_list
        
        #When depth is greater than 0, then find all the neighbours for the current state and recursively call the depth first search 
        elif(depth>0):
            #iterating through each neighbour of current state
            for neighbour in neighbours(current_state):
                #Visit the neighbour only if the neighbour is not visited in the past for the same branch
                if (tuple(neighbour) not in visited):
                    #if the neighbour is not visited, add the neighbour to elements of visited set 
                    visited.add(tuple(neighbour))
                    #recursively call the depth firsh search for the neighbour with depth -1 
                    result=dfs(neighbour,depth-1,path+[neighbour],goal_state)

                    #In Case,calling depth first search with the neighbour as current state leads to goal state
                    if result:
                        #return the result to main function
                        return result
                    #Remove the neighbour from the elements of visited set
                    visited.remove(tuple(neighbour))
            
            #Return Null otherwise 
            return []
    

    #Defining Goal State    
    goal_state=[0,1,2,3,4,5,6,7,8]
    #Declaring a set for storing visited elements of same branch
    visited=set()

    #Iterating for different depths
    for depth_limit in range(1,5):
        #Iteratively calling the Depth First Search for different depths
        result=dfs(puzzle,depth_limit,[puzzle],goal_state)
        
        #In Case, the result returned by the Depth First Search is not Null
        if result:
            #Return the result to the main function
            return result
    
    #Otherwise, return empty list
    return []

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

 # This will be for next project
def astar(puzzle):
    
    goal_state=[0,1,2,3,4,5,6,7,8]

    def heuristic(node):
        total_distance=0
        
        for i in range(0,9):
            a=node.index(i)
            b=goal_state.index(i)
            total_distance= total_distance + abs(int(a/3)-int(b/3)) + abs(a%3 - b%3)
        
        return total_distance
    
    def f(node,g):
        return g + heuristic(node)
    
    open_set=[(heuristic(puzzle),puzzle,[puzzle])] #(Heuristic, Current State, Path)
    closed_set=set()

    while open_set:
        open_set.sort()
        current_node=open_set.pop(0)
        heuristic_value,node,path=current_node

        if(node==goal_state):
            swap_index=[]
            for i in path[1:]:
                swap_index.append(i.index(8))
            return swap_index
        
        closed_set.add(tuple(node))

        for neighbour in neighbours(node):
            if(tuple(neighbour) not in closed_set):
                g_value=len(path) #Cost required to reach the neighbour state
                f_value=f(neighbour,g_value)
                open_set.append((f_value,neighbour,path+[neighbour]))
        
    return []
    


a=iterativeDeepening([0, 4, 1, 3, 8, 2, 6, 7, 5])
print("Iterative Deepning Output:",a)

a=astar([0, 4, 1, 3, 8, 2, 6, 7, 5])
print("A* Algorithm Output:",a)








