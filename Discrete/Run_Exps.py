import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import IPython
import plot_grid
import numpy as np
import scobee
import random
from mdp_utils import calculate_q_values, logsumexp, FP_and_FN_and_TP


nx = 8 # columns
ny = 6 # rows
gamma = 0.95
noise = 0.1
terminal_states = [0]#[rows * columns - 1]
N_exps = 1

for ii in range(N_exps):
	rewards = - np.ones(ny * nx)
	rewards[0] = 2.0
	# rewards[5] = - 1000
	# rewards[6] = - 1000
	# rewards[7] = - 1000
	# constraints is a binary vector indicating hte presence or not of a constraint
	constraints = np.zeros(ny * nx)
	# set some constrains by hand
	# np.random.choice(5, 3, replace=False)
	N_constr = 6
	inds = random.sample(list(range(1, ny * nx)), N_constr)
	# IPython.embed()
	#[16, 17, 24, 25, 23, 31]
	constraints[inds] = 1
	rewards[inds] = -10
	rewards_fix = copy.deepcopy(rewards)
	# num_cnstr = len(np.nonzero(constraints)[0])

	conf_constraints = copy.deepcopy(constraints)

	env2 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_states, rewards, constraints, gamma, noise)
	# np.save('../birl-v2/original', env2.transitions)

	# IPython.embed()
	#visualize rewards
	# print("random rewards")
	# mdp_utils.print_array_as_grid(env2.rewards, env2)
	# print("optimal policy for random grid world")
	# mdp_utils.visualize_policy(mdp_utils.get_optimal_policy(env2), env2)
	optimal_policy = mdp_utils.get_optimal_policy(env2)
	# plot_grid.plot_grid(nx, ny, env2.state_grid_map, constraints, optimal_policy)
	bottom_right_corner = ny * nx - 1
	trajectory_demos = []
	boltz_beta = 1 # large greedy small random


	for i in range(40):
		trajectory_demos.append(mdp_utils.generate_boltzman_demo(env2, boltz_beta, bottom_right_corner))
		# trajectory_demos.append(mdp_utils.generate_optimal_demo(env2, bottom_right_corner))

	trajectory_demosl = [item for sublist in trajectory_demos for item in sublist]
	# IPython.embed()
	env_orig = copy.deepcopy(env2)

	
	
	birl = bayesian_irl.BIRL(env2, trajectory_demosl, boltz_beta, env_orig)

	num_steps = 4000
	step_stdev = 0.1
   
	birl.run_mcmc_bern_constraint(num_steps, 0.5, rewards_fix)
	plot_grid.plot_performance(birl.Perf_list)
	map_constraints = birl.get_map_solution()
	constraints_map = np.zeros(ny * nx)
	
	# acc_rate = birl.accept_rate
	# map_sol, map_rew = birl.get_map_solution()
	# print("Mean results constr", mean_constraints)
	# print("Mean results penalty", mean_penalty)
	# plot_grid.plot_grid_mean_constr(nx, ny, env2.state_grid_map, kk, constraints, mean_constraints, trajectory_demos, optimal_policy)
	# plot_grid.plot_grid(nx, ny, env2.state_grid_map, kk, constraints, trajectory_demos, optimal_policy)
  
	# ============================================================
	cl = scobee.scobee(env2, trajectory_demos)
	# cl.find_k_constraints(6)
	mdp_utils.print_array_as_grid(env2.constraints, env2)
	cl.find_all_constraints()
	# ============================================================

	IPython.embed()
	# print("True Constraints")
	# mdp_utils.print_array_as_grid(env2.constraints, env2)
	# print("=============================================")
	# print("Constraints found by algo")
	# mdp_utils.print_array_as_grid(env2.constraints, env2)
	# TPR, FPR, FNR = FP_and_FN_and_TP(conf_constraints, env2.constraints)
	# print('*******')
	# print(TPR,FPR,FNR)
	# print('*******')
	# TPRt, FPRt, FNRt = FP_and_FN_and_TP(constraints, constraints)
    # Perf_list.append((i,TPR,FPR,FNR))
	# IPython.embed()
   
