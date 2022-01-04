import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import IPython
import plot_grid
import numpy as np

if __name__ == "__main__":

    nx = 24 # columns
    ny = 24 # rows
    gamma = 0.95
    noise = 0.1
    print("create a random ny x nx gridworld with no featurized reward function")
    
    # rewards[5] = - 1000
    # rewards[6] = - 1000
    # rewards[7] = - 1000
    # constraints is a binary vector indicating hte presence or not of a constraint
    constraints = np.zeros(ny * nx)
    rewards = - np.ones(ny * nx)
    # set some constrains by hand
    inds = [ 3, 4, 5, nx +3, nx+4,nx+5, 3*nx+int(np.floor(3*ny/4)),3*nx+int(np.floor(3*ny/4))+1,3*nx+int(np.floor(3*ny/4))+2,
    4*nx+int(np.floor(3*ny/4)),4*nx+int(np.floor(3*ny/4))+1,4*nx+int(np.floor(3*ny/4))+2,
    4*nx,4*nx+1,5*nx,5*nx+1,6*nx+int(np.floor(2*ny/4)),7*nx+int(np.floor(2*ny/4))-1,7*nx+int(np.floor(2*ny/4)),7*nx+int(np.floor(2*ny/4))+1,
    8*nx+int(np.floor(2*ny/4)),(int(np.floor(4*nx/5))-1)*nx+int(np.floor(4*ny/5)),(int(np.floor(4*nx/5))-1)*nx+int(np.floor(4*ny/5))+1,
    (int(np.floor(4*nx/5)))*nx+int(np.floor(4*ny/5))+1,
    (int(np.floor(4*nx/5))+1)*nx+int(np.floor(4*ny/5))+1,(int(np.floor(4*nx/5))+2)*nx+int(np.floor(4*ny/5))+1,
    (int(np.floor(4*nx/5))+2)*nx+int(np.floor(4*ny/5)),
    int(np.floor(3*nx/4))*nx+int(np.floor(1*ny/5))-1,int(np.floor(3*nx/4))*nx+int(np.floor(1*ny/5)),int(np.floor(3*nx/4))*nx+int(np.floor(1*ny/5))+1,
    (int(np.floor(3*nx/4))+1)*nx+int(np.floor(1*ny/5))-1,(int(np.floor(3*nx/4))+1)*nx+int(np.floor(1*ny/5)),(int(np.floor(3*nx/4))+1)*nx+int(np.floor(1*ny/5))+1,
    (int(np.floor(3*nx/4))+2)*nx+int(np.floor(1*ny/5)), (int(np.floor(5*nx/8))+1)*nx+int(np.floor(5*ny/8))+2,
    (int(np.floor(5*nx/8))+1)*nx+int(np.floor(5*ny/8))+1,(int(np.floor(5*nx/8))+2)*nx+int(np.floor(5*ny/8))+1,
    (int(np.floor(5*nx/8))+3)*nx+int(np.floor(5*ny/8))+1,(int(np.floor(5*nx/8))+4)*nx+int(np.floor(5*ny/8))+1,
    (int(np.floor(5*nx/8))+5)*nx+int(np.floor(5*ny/8))+1,
    (int(np.floor(5*nx/8))+5)*nx+int(np.floor(5*ny/8))+2]

    constraints[inds] = 1
    

    rewards[inds] = -10

    num_cnstr = len(np.nonzero(constraints)[0])
         
    env2 = mdp_worlds.nonrand_gridworld(ny, nx, [], rewards, constraints, gamma, noise)
    # np.save('../birl-v2/original', env2.transitions)
    constraints_array = np.zeros((nx, ny))

    for i in inds:
        temp = env2.state_grid_map[i]
        constraints_array[ny - 1 - temp[1], temp[0]] = 1
    # IPython.embed()   
    #visualize rewards
    # print("random rewards")
    # mdp_utils.print_array_as_grid(env2.rewards, env2)
    # print("optimal policy for random grid world")
    # mdp_utils.visualize_policy(mdp_utils.get_optimal_policy(env2), env2)
    # optimal_policy = mdp_utils.get_optimal_policy(env2)
    # plot_grid.plot_grid(nx, ny, env2.state_grid_map, constraints, optimal_policy)
    trajectory_demos = []
    boltz_beta = 1
    # env_orig = copy.deepcopy(env2)
    perf_lists = []
    trajectory_demos = []
    nx_temp = int(nx/2)
    ny_temp = int(ny/2)
    # list_inds = [list(range(nx_temp)), list(range(nx_temp, nx)), 
    # list(range(nx * nx_temp, nx * nx_temp + nx_temp)), list(range(nx * nx_temp + nx_temp,nx * (nx_temp + 1)))]
    list_inds = [(0,0),(0,nx_temp),(ny_temp,0), (ny_temp, nx_temp)]
    terminal_states = [0, nx_temp -1, nx_temp * ny_temp - 1, 0]#[rows * columns - 1]
    start_states = [nx_temp * ny_temp - 1, nx_temp * (ny_temp - 1), 0, nx_temp * ny_temp - 1]

    mean_penalty_list = []
    
    for N_demos in [20, 50, 100, 200]:

        for hier in range(4):
        
            print('Now running {} and {}'.format(N_demos, hier))

            rewards_temp = - np.ones(ny_temp * nx_temp)
            # rewards negative everywhere except for the goal state
            rewards_temp[terminal_states[hier]] = 2.0
            # IPython.embed()
            constraints_temp = constraints_array[list_inds[hier][0]:list_inds[hier][0] + ny_temp, list_inds[hier][1]:list_inds[hier][1] + nx_temp]
            inds_temp = np.nonzero(constraints_temp.flatten())
            constraints_temp = np.zeros(ny_temp * nx_temp)
            constraints_temp[inds_temp] = 1


            rewards_temp[inds_temp] = -10
            env2 = mdp_worlds.nonrand_gridworld(ny_temp, nx_temp, [terminal_states[hier]], rewards_temp, constraints_temp, gamma, noise)
            # np.save('../birl-v2/original', env2.transitions)
            
            
            optimal_policy = mdp_utils.get_optimal_policy(env2)
            # plot_grid.plot_grid(nx, ny, env2.state_grid_map, constraints, optimal_policy)
            bottom_right_corner = start_states[hier]
            boltz_beta = 1
            env_orig = copy.deepcopy(env2)
            perf_lists = []
            trajectory_demos = []


            
            for i in range(N_demos):
                # trajectory_demos.append(mdp_utils.generate_optimal_demo(env2, bottom_left_corner))
                trajectory_demos.append(mdp_utils.generate_boltzman_demo(env2, boltz_beta, bottom_right_corner))
            trajectory_demos = [item for sublist in trajectory_demos for item in sublist]




            # IPython.embed()

            plot_grid.plot_grid(nx_temp, ny_temp, env2.state_grid_map, hier, N_demos, constraints_temp, trajectory_demos)
            # IPython.embed()
            for kk in range(1):

                print('Iteration {}'.format(kk))
               
                print("Running Bayesian IRL with full optimal policy as demo")
                demos = mdp_utils.demonstrate_entire_optimal_policy(env2)
               
                rewards_fix = - np.ones(ny_temp * nx_temp)
                rewards_fix[terminal_states[hier]] = 2.0
                # birl.run_mcmc_bern(num_steps, 0.5, rewards_fix)
                # map_constr, map_rew = birl.get_map_solution()
                constraints_map = np.zeros(ny * nx)
                

                # constraints[[16, 17, 24, 25, 23, 31]] = 1
              
                birl = bayesian_irl.BIRL(env2, trajectory_demos, boltz_beta, env_orig)

                num_steps = 4000
                step_stdev = 0.1
               
                birl.run_mcmc_bern_constraint(num_steps, 0.5, rewards_fix, hier, N_demos)

                # plot_grid.plot_performance(birl.Perf_list)
                map_constraints = birl.get_map_solution()
                constraints_map = np.zeros(ny * nx)
              
                acc_rate = birl.accept_rate
                mean_constraints, mean_penalty = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
                # print("mean constraints", mean_constraints)
                map_constraints, map_rew = birl.get_map_solution()
                
                np.savetxt('plots/penalty_rew_' + str(N_demos) + '_' + str(hier) + '.txt', birl.chain_rew)
                # IPython.embed()
                temp_query_state = birl.compute_variance(birl.posterior, 0.1)
                # temp_query_state = np.random.randint(nx * ny)
                print(temp_query_state)

                mean_penalty_list.append(mean_penalty)
                np.savetxt('plots/mean_constraints_hier_' + str(N_demos) + '_' + str(hier) + '.txt', mean_constraints)
                np.savetxt('plots/map_constraints_hier_' + str(N_demos) + '_' + str(hier) + '.txt', map_constraints)
                # IPython.embed()
                # if query_state != None:
                #     temp_query_state = list(birl.env.state_grid_map.keys())[list(birl.env.state_grid_map.values()).index(query_state)]
                # print('******',temp_query_state,'*****')
                # if temp_query_state != None:# and temp_query_state != 0 and temp_query_state not in env_orig.constraints:
                #     # temp_query_state = list(self.env.state_grid_map.keys())[list(self.env.state_grid_map.values()).index(query_state)]
                
                # for _ in range(8):
                #     traj = mdp_utils.generate_boltzman_demo(env_orig, boltz_beta, temp_query_state)
                #     query_state_action = traj[0]
                #     trajectory_demos.append(query_state_action)
            # perf_lists.append(birl.Perf_list)
        np.savetxt('plots/mean_reward_penalty_hier_' + str(N_demos) + '.txt', np.array(mean_penalty_list))
    IPython.embed()
       