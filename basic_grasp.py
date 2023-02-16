import random
import graph as gr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


#--------------------------------------------reactive_grasp_for_MiMALBP-2-----------------------------------------------
def grasp(tasks_list, alpha, max_iterations, max_search, n_workstations):

    optimal_value   = 1000                  #best cycle time
    best_solution   = []                    #best solution (assignment )
   
    alpha1_sol      = []
   


  

    for j in range(max_iterations):

        
        #select an alpha value using probabilities
        alpha           = alpha

        #build feasible solutions (not necessarly optimal)
        solution        = construction_phase(alpha, tasks_list)

        #searching for optimal solution using neighborhood search
        solution        = local_search_phase(solution, max_search)

        cy_time         = objective_function(solution, n_workstations)

      
        #comparing soltuion in terms of objective value---------------
        if( cy_time < optimal_value):

            best_solution       = solution
            optimal_value       = cy_time
            
        #-------------------------------------------------------------
        
    


        if( j % 100 == 0):
            print(j)
        
        if(solution not in alpha1_sol):
                alpha1_sol.append(solution)
               
        #-------------------------------------------------------------

    print("cycle_time:", optimal_value)

    print("-----------------------------------------")
    print("number of solutions a1:",len(alpha1_sol))
    #print("p1:", prob1)
    #print("p2:", prob2)
    #print("p3:", prob3)
    print("-----------------------------------------")


    #print(alpha3_sol)
    
    return best_solution, alpha1_sol

#--------------------------------------------------------------------------------------------------------------

#construction phase function
def construction_phase(alpha, tasks_list):
    
    solution        = []
    cl              = {}
    RCL             = {}

    #initialization of the pseudo number generator
    #random.seed(a=seed, version=2)

    #Create RPW_list
    RPW_list        = RPW(tasks_list)

    #associate new weights for each element in alpha
    for task in RPW_list:
        RPW_list[task]      = 1/RPW_list[task]

    while(RPW_list):

        #create the candidate list
        for task in RPW_list:

            pred_list       = np.array(list(gr.G.predecessors(int(task))), dtype=str)

            if not any(item in  pred_list for item in RPW_list):
                    cl[task]        =  RPW_list[task]

       
        weights_cl      = cl.values()
        #calculate the threshold value
        threshold       = min(weights_cl) + alpha*(max(weights_cl) - min(weights_cl))
        
        #create Restricted candidate list
        for task in cl:

            if(cl[task] <= threshold):
                RCL[task]       = cl[task]

    
        #select randomly a task from the restricted candidate list
        selected_task       = random.choice(list(RCL.keys()))

        #addind task to the partial solution
        solution.append(selected_task) 

        #delete selected task from cl and RPW_list
        RPW_list.pop(selected_task)
        #cl.pop(selected_task)
        cl.clear()
        #clean the RCL
        RCL.clear()

    return  list(np.array(solution, dtype=int))

#local search function
def local_search_phase(solution, max_search):
    
    neighbor_solutions       = []
    neighbor      = []
    best_solution           = solution
    iteraion_crit       = 0
    

    for j in range(max_search):

            while(True):
                
                sol_copy      = solution.copy()
                #search for neighbor solution
                neighbor      = random_swap(sol_copy)

                if(not list(neighbor) in neighbor_solutions):

                        neighbor_solutions.append(list(neighbor))
               
                        break

                iteraion_crit       = iteraion_crit + 1
                
                if(iteraion_crit > 30):
                        break
           
    for i in neighbor_solutions:
    #compare the neighbor solution with the old solution(constructed solution)
        if(objective_function(i, n_workstations) < objective_function(best_solution, n_workstations)):

            best_solution       = i


    return  best_solution

#objective function to evaluat found soutions
def objective_function(solution, n_workstations):
    
    cycle_time              = 0
    initial_cycle_time      = 0
    number_wr               = 1
    
    int_solution        = np.array(solution, dtype=int)

    
    #initialization of the cycle time with highest task time
    for task in int_solution:

        if(gr.G.nodes[task]["task"+str(task)] > initial_cycle_time):

            initial_cycle_time = gr.G.nodes[task]["task"+str(task)]
    #---------------------------------------------------------------

    

    while(True):
    
        workstation_time        = initial_cycle_time
       
       
        
        for task in int_solution:

            if(gr.G.nodes[task]["task"+str(task)] <= workstation_time):

                workstation_time        = workstation_time - gr.G.nodes[task]["task"+str(task)]

            elif(gr.G.nodes[task]["task"+str(task)] > workstation_time):

                workstation_time        = initial_cycle_time
                number_wr               = number_wr + 1
                workstation_time        = workstation_time - gr.G.nodes[task]["task"+str(task)]
                #print("current nb",number_wr)
                #input()

        if(number_wr > n_workstations):

            
            initial_cycle_time      = initial_cycle_time + 0.1
            number_wr               = 1   
        
        else:

            cycle_time      = initial_cycle_time
            break

    
    
    return cycle_time

#Ranked Positional Weight
def RPW(tasks_list2):
    
    RPW_list            = {}
    highest_pw          = 0
    positional_weight   = 0
    tasks_list          = tasks_list2.copy()

    #initialize the highest_pw by the PW of the fisrt element in tasks_list--------
    #---create the list of successors of the first element

    

    while(tasks_list):
        
        for task in tasks_list:

                suc_list        = gr.nx.nodes(gr.nx.dfs_tree(gr.G, task))

                for i in suc_list:

                    positional_weight       = positional_weight + gr.G.nodes[i]["task"+str(i)]
                    


                if(positional_weight > highest_pw):

                    highest_pw      = positional_weight
                    t               = task
                
                positional_weight       = 0
       
        #RPW_list.append(t)
        RPW_list[str(t)]        = highest_pw
        tasks_list.remove(t)
        
        highest_pw      = 0
    #------------------------------------------------------------------------------

    


    return RPW_list

def random_swap(solution):
    
    #neighbor_solution       = []
 
    while(True):

        task1       = random.choice(solution)
        index1      = solution.index(task1)

        
        task2       = random.choice([task for task in solution if task != task1])
        index2      = solution.index(task2)
        

        x           = index1
        y           = index2
        key         = True

        if( index1 < index2):
                            
                        if(task1 not in gr.nx.nodes(gr.nx.bfs_tree(gr.G, task2, reverse=True))):
                                
                                for a in solution[index1+1:index2]:
                                    
                                    if(task1 in gr.nx.nodes(gr.nx.bfs_tree(gr.G, a, reverse=True)) or a in gr.nx.nodes(gr.nx.bfs_tree(gr.G, task2, reverse=True))):
                                        key     = False
                                        continue

                                if(key == True):
                                    
                                    solution[x]     = task2
                                    solution[y]     = task1
                                    break
                                

        else:

                            if(task2 not in gr.nx.nodes(gr.nx.bfs_tree(gr.G, task1, reverse=True))):
                            
                                for a in solution[index2+1:index1]:
                                    
                                    if(task2 in gr.nx.nodes(gr.nx.bfs_tree(gr.G, a, reverse=True)) or a in gr.nx.nodes(gr.nx.bfs_tree(gr.G, task1, reverse=True)) ):
                                        key     = False
                                        break
                                
                                if(key == True):
                                    
                                    solution[x]     = task2
                                    solution[y]     = task1
                                    break




    return solution

    


#tasks_list      =[1,2,3,4,5,6,7,8,9,10,11,12]
#tasks_list      = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
tasks_list      = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
alpha           = 0
max_iterations  = 2000
max_search      = 35
n_workstations  = 6

start_time = datetime.now()
solution, s1= grasp(tasks_list, alpha, max_iterations,max_search, n_workstations)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

print(solution)


"""
sol     = construction_phase(1, tasks_list)
sol     = list(np.array(sol, dtype=int))

print(sol)
print(sol[2:4])
"""
#print(random_swap(sol))
#print(gr.nx.nodes(gr.nx.bfs_tree(gr.G, 5, reverse=True)))
#print("its cycle time:", objective_function(sol, 3))
"""
list1 = RPW(tasks_list)
#print(list1)
cl = {}
for task in list1:

    print(list(gr.G.predecessors(int(task))))
    pred_list       = np.array(list(gr.G.predecessors(int(task))), dtype=str)

    #any(item in list(gr.G.predecessors(a)) for item in CL)
    if not any(item in  pred_list for item in list1):
        cl[task]        =  list1[task]

print(random.choice(list(cl.keys())))
#print(random.choice(cl.ite))

"""
