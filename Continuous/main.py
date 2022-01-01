import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import IPython
import plot_grid
import numpy as np
from Q_agent_trajectories import DQLAgent_traj
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # nx = 8 # columns
    # ny = 6 # rows
    # gamma = 0.95
    # noise = 0.1
    # terminal_states = [0]#[rows * columns - 1]
    # rewards negative everywhere except for the goal state
    # rewards = - np.ones(ny * nx)
    # rewards[0] = 2.0
    # rewards[5] = - 1000
    # rewards[6] = - 1000
    # rewards[7] = - 1000
    # constraints is a binary vector indicating hte presence or not of a constraint
    # constraints = np.zeros(ny * nx)
    # # set some constrains by hand
    # constraints[[16, 17, 24, 25, 23, 31]] = 1
    # rewards[[16, 17, 24, 25, 23, 31]] = -10
    # num_cnstr = len(np.nonzero(constraints)[0])

    # env2 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_states, rewards, constraints, gamma, noise)
    envtest = mdp_worlds.Cont2D_v2(disc_x=40, disc_y=40, disc_act_x=8, disc_act_y=3)
    # IPython.embed()
    # for i in range(14,15):#envtest.num_states
    #     for j in range(envtest.num_actions):
    #         ind = int(np.nonzero(envtest.transitions[i][j])[0])
    #         print(envtest.state_grid_map[i], envtest.state_grid_map[ind], envtest.action_grid_map[j])
    

    # temp_env = DQLAgent_traj()

    V = mdp_utils.value_iteration(envtest)
    # IPython.embed()
    # np.savetxt('V_values', V, delimiter=',')
    # V = np.loadtxt('V_values',delimiter=',')

    # plot_grid.plot_grid_test(10, 10, envtest.state_grid_map, V)
    
    # for i in range(len(envtest.num_state_grid_map)):
    #     print(envtest.num_state_grid_map[i],V[i])

    boltz_beta = 2
    bottom_right_corner = envtest.state_grid_map[(0.5,0.0)]
    temp_traj = []
    for i in range(6):
        traj = mdp_utils.generate_boltzman_demo(envtest, V, boltz_beta, bottom_right_corner)
        temp_traj.append(traj)
        # cnt = 0
        # new_list = []
        # for i in traj:
        #     # print(envtest.num_state_grid_map[i[0]])
        #     new_list.append((envtest.num_state_grid_map[i[0]]))
        # new_list = np.array(new_list)
        # plt.plot(new_list[:,0],new_list[:,1])
        # plt.show()
    plot_grid.plot_cont_2D_v2(envtest, temp_traj, start=[0.5,0], goal=[0.5,1], constr=[0.5,0.5], r_constr = 0.15, r_goal = 0.1)

    trajector = [item for sublist in temp_traj for item in sublist]
    # IPython.embed()
    cnt = 0
    new_list = []
    for i in trajector:
        # print(envtest.num_state_grid_map[i[0]])
        new_list.append((envtest.num_state_grid_map[i[0]], envtest.num_action_grid_map[i[1]]))
        # temp[cnt][0]=envtest.num_state_grid_map[i[0]][0]
        # temp[cnt][1]=envtest.num_state_grid_map[i[0]][1]
        cnt += 1
    trajectory = new_list
    # plt.plot(temp[:,0],temp[:,1])
    # plt.show()

    # IPython.embed()
    # traj = temp_env.obtain_boltz_trajectories()
    # trajectories = []
    # for i in range(40):
    #     # trajectory = mdp_utils.generate_noisy_demo(envtest, 0, [0.2,0.1])generate_weird_demo(env, start_state)
    #     trajectory = mdp_utils.generate_weird_demo(envtest,[1, 0.0])

    #     # IPython.embed()
    #     # plot_grid.plot_cont_2D(trajectory, envtest.goal)
    #     trajectories.append(trajectory)

 
    # plot_grid.plot_cont_2D(trajectory, envtest.goal)
    
    # plot_grid.plot_cont_2D(trajectory, envtest.goal)

    # trajectory = [item for sublist in traj for item in sublist]
    
    # plot_grid.plot_cont_2D(trajectory, envtest.goal)

    envtest = mdp_worlds.Cont2D_v2(10, 10, 8, 3)
    # IPython.embed()
    discretized_trajectory = mdp_utils.discretize_traj(envtest, trajectory, 0.1, envtest.max_speed)

    # trajectory_demos = [item for sublist in trajectories for item in sublist]
    # IPython.embed()
    # envtest.state_grid_map
    disc_traj = []
    # IPython.embed()
    for demo in discretized_trajectory:
        key_state=None
        key_act=None
        for key_temp, val in envtest.num_state_grid_map.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if np.array_equal(val, demo[0:2]):
                key_state = key_temp
                # IPython.embed()
        for key_temp, val in envtest.num_action_grid_map.items(): 
            if np.array_equal(val, demo[2:]):
                key_act = key_temp
        if key_state==None:
            IPython.embed()
        disc_traj.append((key_state,key_act))
    # IPython.embed()



    
    
    
    ################################################################################################
    # np.save('../birl-v2/original', env2.transitions)
    
    # IPython.embed()
    #visualize rewards
    # print("random rewards")
    # mdp_utils.print_array_as_grid(env2.rewards, env2)
    # print("optimal policy for random grid world")
    # mdp_utils.visualize_policy(mdp_utils.get_optimal_policy(env2), env2)
    # optimal_policy = mdp_utils.get_optimal_policy(env2)
    # plot_grid.plot_grid(nx, ny, env2.state_grid_map, constraints, optimal_policy)
    # bottom_right_corner = ny * nx - 1
    trajectory_demos = []
    boltz_beta = 2
    env_orig = copy.deepcopy(envtest)
    # perf_lists = []
    trajectory_demos = disc_traj
    # for i in range(4):
    #     # trajectory_demos.append(mdp_utils.generate_optimal_demo(env2, bottom_left_corner))
    #     trajectory_demos.append(mdp_utils.generate_boltzman_demo(env2, boltz_beta, bottom_right_corner))
    # trajectory_demos = [item for sublist in trajectory_demos for item in sublist]
    for kk in range(1):

        print('Iteration {}'.format(kk))

        print("Running Bayesian IRL with full optimal policy as demo")
        # demos = mdp_utils.demonstrate_entire_optimal_policy(env2)
        # IPython.embed()
        rewards_fix = -np.ones(len(envtest.rewards))
        for i in envtest.terminals:
            rewards_fix[i] = 2


        # rewards_fix[0] = 2.0
        # birl.run_mcmc_bern(num_steps, 0.5, rewards_fix)
        # map_constr, map_rew = birl.get_map_solution()
        # constraints_map = np.zeros(ny * nx)
        

        # constraints[[16, 17, 24, 25, 23, 31]] = 1
        # IPython.embed()
        birl = bayesian_irl.BIRL(envtest, trajectory_demos, boltz_beta, env_orig)
        

        num_steps = 4000
        step_stdev = 0.1
        birl.run_mcmc_bern_constraint(num_steps, 0.5, rewards_fix)
        # IPython.embed()
        # plot_grid.plot_performance(birl.Perf_list)
        map_constraints = birl.get_map_solution()
        # constraints_map = np.zeros(ny * nx)
      
        acc_rate = birl.accept_rate
        # constraints[[16, 17, 24, 25, 23, 31]] = 1
        mean_cnstr = birl.get_mean_solution(burn_frac=0.1, skip_rate=1)
        # plot_grid.plot_posterior(nx, ny, mean_cnstr, kk, constraints, True, env2.state_grid_map)
        # print("map constraints", map_constraints)
        mean_constraints, mean_penalty = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
        # print("mean constraints", mean_constraints)
        map_sol, map_rew = birl.get_map_solution()
        

        # IPython.embed()
        # plot_grid.plot_grid_mean_constr(nx, ny, env2.state_grid_map, kk, constraints, mean_constraints, trajectory_demos, optimal_policy)
        plot_grid.plot_grid_mean_constr(envtest.disc_x + 1, envtest.disc_y + 1, envtest.num_state_grid_map, kk, mean_constraints, mean_constraints)

        # plot_grid.plot_grid_temp2(nx, ny, env2.state_grid_map, constraints_plot, constraints_map, trajectory_demos, optimal_policy, kk)
        np.savetxt('plots/penalty_rew' + str(kk) + '.txt', birl.chain_rew)
        # IPython.embed()
        temp_query_state = birl.compute_variance(birl.posterior, 0.1)
        print(temp_query_state)
        IPython.embed()
        # if query_state != None:
        #     temp_query_state = list(birl.env.state_grid_map.keys())[list(birl.env.state_grid_map.values()).index(query_state)]
        # print('******',temp_query_state,'*****')
        # if temp_query_state != None:# and temp_query_state != 0 and temp_query_state not in env_orig.constraints:
        #     # temp_query_state = list(self.env.state_grid_map.keys())[list(self.env.state_grid_map.values()).index(query_state)]
        
        # for _ in range(4):
        #     traj = mdp_utils.generate_boltzman_demo(env_orig, boltz_beta, temp_query_state)
        #     query_state_action = traj[0]
        #     trajectory_demos.append(query_state_action)
        # perf_lists.append(birl.Perf_list)

    # temp_perf_list = []
    # for i in range(len(perf_lists[0])):
    #     tempTPR = 0
    #     tempFPR = 0
    #     tempFNR = 0

    #     for j in range(len(perf_lists)):
    #         #                                                               IPython.embed()
    #         tempTPR += perf_lists[j][i][1]
    #         tempFPR += perf_lists[j][i][2]
    #         tempFNR += perf_lists[j][i][3]


    #     temp_perf_list.append((i,tempTPR/len(perf_lists),tempFPR/len(perf_lists),tempFNR/len(perf_lists)))

    # plot_grid.plot_performance(temp_perf_list)




# #-----------------------------------    
#     print()
#     print("create simple featurized mdp")
#     env = mdp_worlds.gen_simple_world()
#     #get optimal policy 
#     opt_pi = mdp_utils.get_optimal_policy(env)
#     #text-based visualization of optimal policy
#     print("optimal policy for featurized grid world") 
#     mdp_utils.visualize_policy(opt_pi, env)


# #-----------------------------------
#     print()
#     print("generate a demonstration starting from top right state (states are numbered 0 through num_states")
#     opt_traj = mdp_utils.generate_optimal_demo(env, 2)
#     print("optimal trajectory starting from state 2")
#     print(opt_traj)
#     #format with arrows
#     mdp_utils.visualize_trajectory(opt_traj, env)

# #-----------------------------------
#     print()
#     print("Bayesian IRL")
#     #give an entire policy as a demonstration to Bayesian IRL (best case scenario, jjst to test that BIRL works)
#     print("Running Bayesian IRL with full optimal policy as demo")
#     demos = mdp_utils.demonstrate_entire_optimal_policy(env)
#     print(demos)
#     beta = 10.0 #assume near optimal demonstrator
#     birl = bayesian_irl.BIRL(env, demos, beta)
#     num_steps = 2000
#     step_stdev = 0.1
#     birl.run_mcmc(num_steps, step_stdev)
#     map_reward = birl.get_map_solution()
#     print("map reward", map_reward)
#     mean_reward = birl.get_mean_solution(burn_frac=0.1,skip_rate=2)
#     print("mean reward", mean_reward)

#     # visualize the optimal policy for the learned reward function
#     env_learned = copy.deepcopy(env)
#     env_learned.set_rewards(map_reward)
#     learned_pi = mdp_utils.get_optimal_policy(env_learned)
#     #text-based visualization of optimal policy
#     print("Learned policy from Bayesian IRL using MAP reward") 
#     mdp_utils.visualize_policy(learned_pi, env_learned)
#     print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
#     #we could also implement an adaptive step MCMC but this vanilla version shoudl suffice for now

#     #-----------------------------------
#     print()
#     ## Quantitative metrics
#     print("policy loss", mdp_utils.calculate_expected_value_difference(learned_pi, env))
#     print("policy action accuracy {}%".format(mdp_utils.calculate_percentage_optimal_actions(learned_pi, env) * 100))


#     #-----------------------------------
#     print()
#     ## Bayesian IRL with a demonstration purposely chosen to not be super informative
#     print("running Bayesian IRL with the following demo")
#     opt_traj = mdp_utils.generate_optimal_demo(env, 7)
#     print("optimal trajectory starting from state 7")
#     print(opt_traj)
#     #format with arrows
#     mdp_utils.visualize_trajectory(opt_traj, env)


#     birl = bayesian_irl.BIRL(env, opt_traj, beta)
#     num_steps = 2000
#     step_stdev = 0.1
#     birl.run_mcmc(num_steps, step_stdev)
#     map_reward = birl.get_map_solution()
#     print("map reward", map_reward)
#     mean_reward = birl.get_mean_solution(burn_frac=0.1,skip_rate=2)
#     print("mean reward", mean_reward)

#     # visualize the optimal policy for the learned reward function
#     env_learned = copy.deepcopy(env)
#     env_learned.set_rewards(map_reward)
#     learned_pi = mdp_utils.get_optimal_policy(env_learned)
#     #text-based visualization of optimal policy
#     print("Learned policy from Bayesian IRL using MAP reward") 
#     mdp_utils.visualize_policy(learned_pi, env_learned)
#     print("accept rate for MCMC", birl.accept_rate) #good to tune number of samples and stepsize to have this around 50%
#     #we could also implement an adaptive step MCMC but this vanilla version shoudl suffice for now

#     #-----------------------------------
#     print()
#     ## Quantitative metrics
#     print("policy loss", mdp_utils.calculate_expected_value_difference(learned_pi, env))
#     print("policy action accuracy {}%".format(mdp_utils.calculate_percentage_optimal_actions(learned_pi, env) * 100))

#     #demo isn't super informative so it doesn't learn the true optimal policy but does learn that the red feature is probably negative
#     # and likely worse than the white feature

#     print("learned weights", map_reward)
#     print("true weights", env.feature_weights)



