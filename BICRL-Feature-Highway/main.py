import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import IPython
import plot_grid
import numpy as np

if __name__ == "__main__":

    nx = 3 # columns
    ny = 11 # rows
    gamma = 0.95
    noise = 0.1
    print("create a random ny x nx gridworld with no featurized reward function")
    # rewards negative everywhere except for the goal state
    terminal_states = [0]#[rows * columns - 1]
    # rewards negative everywhere except for the goal state
    rewards = - np.zeros(ny * nx)
    # rewards[terminal_states] = 2.0
    
    # constraints is a binary vector indicating hte presence or not of a constraint
    constraints = np.zeros(ny * nx)# not used
    
    rewards_fix = copy.deepcopy(rewards)
    env2 = mdp_worlds.highway(ny, nx, terminal_states, rewards, constraints, gamma, noise)
    # np.save('../birl-v2/original', env2.transitions)
    # original feature weights
    W_real = np.zeros(env2.num_features)
    W_real[0:3] = -5.0
    W_real[3] = 0.5
    W_real[4:7] = -5.0
    W_real[7] = -5.0
    # W_real[-2] = 1
    W_real[8] = 5.0

    W_fix = np.zeros(env2.num_features)
    W_fix[3] = 0.5
    W_fix[8] = 5.0

    real_rewards = env2.get_rewards(W_real, 0, rewards_fix)
    env2.set_rewards(real_rewards)
    # bottom_right_corner = ny * nx - 1
    mean_penalty_list = []
    N_demonstrations = 100
    # gather trajectories
    for kk in range(100):
        trajectory_demos = []
        boltz_beta = 1
        env_orig = copy.deepcopy(env2)
        perf_lists = []
        trajectory_demos = []
        bottom_right_corner = env2.num_states - 2
        for i in range(N_demonstrations):
            # trajectory_demos.append(mdp_utils.generate_optimal_demo(env2, bottom_left_corner))
            trajectory_demos.append(mdp_utils.generate_boltzman_demo(env2, boltz_beta, bottom_right_corner))
        trajectory_demos = [item for sublist in trajectory_demos for item in sublist]

       
        plot_grid.plot_grid(nx, ny, env2.state_grid_map, 0, 0, constraints, trajectory_demos)

        print('Iteration {}'.format(kk))
       
        print("Running Bayesian IRL with full optimal policy as demo")
       
       
        constraints_map = np.zeros(ny * nx)
                  
        birl = bayesian_irl.BIRL(env2, trajectory_demos, boltz_beta)

        num_steps = 2000
        step_stdev = 0.1
       
        birl.run_mcmc_bern_constraint(num_steps, rewards_fix)

        # plot_grid.plot_performance(birl.Perf_list)
        # map_constraints = birl.get_map_solution()
        # constraints_map = np.zeros(ny * nx)
      
        acc_rate = birl.accept_rate
        
        mean_constraints = birl.get_mean_solution(burn_frac=1500, skip_rate=1)
        # IPython.embed()
        # print("mean constraints", mean_constraints)
        # map_constraints, map_rew = birl.get_map_solution()
        
        # mean_penalty_list.append(mean_penalty)
        # name = 'feat_mean_cnstr_' + str(kk)
        # plot_grid.plot_grid_mean_constr(nx, ny, env2.state_grid_map, kk, N_demos, name, constraints, mean_constraints, trajectory_demos, optimal_policy)
        
        # plot_grid.plot_grid_temp2(nx, ny, env2.state_grid_map, constraints_plot, constraints_map, trajectory_demos, optimal_policy, kk)
        # np.savetxt('plots/penalty_rew_single_' + str(N_demos)+'_'+str(kk) + '.txt', birl.chain_rew)


        np.savetxt('plots/feat_mean_constraints_single_' + str(kk) + '.txt', mean_constraints)
        # np.savetxt('plots/map_constraints_single_' + str(N_demos)+'_'+str(kk) + '.txt', map_constraints)
      




