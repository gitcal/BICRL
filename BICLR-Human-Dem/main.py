import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import IPython
import plot_grid
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":


    # load human demos
    data_traj = []
    for i in range(2,22):
        temp = np.loadtxt('data/t'+str(i)+'.txt', delimiter=',')
        data_traj.append(temp)
    env = mdp_worlds.Cont2D_v2(disc_x=11, disc_y=11, disc_act_x=8, disc_act_y=4, noise = 0.05)

    data_traj = env.get_actions_from_human_demos(data_traj) 
    trajector = [item for sublist in data_traj for item in sublist]    

    # discretize demos
    disc_traj = mdp_utils.discretize_traj(env, trajector)

    disc_traj2 = []
    for demo in disc_traj:
        key_state=None
        key_act=None
        for key_temp, val in env.num_state_grid_map.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if np.array_equal(val, demo[0:2]):
                key_state = key_temp
        for key_temp, val in env.num_action_grid_map.items(): 
            if np.array_equal(val, demo[2:]):
                key_act = key_temp
        if key_state==None:
            IPython.embed()
        disc_traj2.append((key_state,key_act))


    boltz_beta = 10
    env_orig = copy.deepcopy(env)
    trajectory_demos = disc_traj2
    
    for kk in range(1):

        print('Iteration {}'.format(kk))

        print("Running Bayesian IRL with full optimal policy as demo")
        # set nominal rewards
        rewards_fix = -np.ones(len(env.rewards))
        for i in env.terminals:
            rewards_fix[i] = env.target_rew
      
        # prior constraint probability
        prior = 0.05
        birl = bayesian_irl.BIRL(env, trajectory_demos, boltz_beta, env_orig, prior)
        
        # run BICRL
        num_steps = 6000
        step_stdev = 0.1
        birl.run_mcmc_bern_constraint(num_steps, 1, rewards_fix)
        
        map_constraints = birl.get_map_solution()
      
        acc_rate = birl.accept_rate
        mean_cnstr = birl.get_mean_solution(burn_frac=0.1, skip_rate=1)
        mean_constraints, mean_penalty = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
        map_sol, map_rew = birl.get_map_solution()    

      
        plot_grid.plot_grid_mean_constr(env.disc_x + 1, env.disc_y + 1, env.num_state_grid_map, kk, mean_constraints, mean_constraints)
       
        np.savetxt('plots/penalty_rew' + str(kk) + '.txt', birl.chain_rew)
        temp_query_state = birl.compute_variance(birl.posterior, 0.1)
        print(temp_query_state)
