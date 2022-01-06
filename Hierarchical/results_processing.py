import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import IPython
import plot_grid
import numpy as np
from mdp_utils import calculate_q_values, logsumexp, FP_and_FN_and_TP

def eval_policy(env_orig, rewards, constraints):


	env_test = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards, constraints, gamma, noise)
	# test_policy = mdp_utils.get_optimal_policy(env_test)

	
	
	tot_rew_list = [] 
	for i in range(20):
		print(i)
		# init_state = env_orig.num_states - 1#np.random.randint(env_orig.num_states)
		# rew_list = []
		# while True:
		# 	state = init_state
		# 	probs = env_orig.transitions[state][test_policy[state]][:]
		# 	state_next = np.random.choice(list(range(env_orig.num_states)), 1, p=probs)
		# 	print(state_next)
		# 	# state_next, reward, done = env_orig.step(test_policy(state), env_orig.state_grid_map[state])
		# 	rew_list.append(env_orig.rewards[state_next])
		# 	if state_next == 0:
		# 		break
		# 	state = state_next
		traj = mdp_utils.generate_boltzman_demo(env_test, boltz_beta, env_orig.num_states - 1)
		tot_rew = 0
		cnt = 0
		for j in traj:
			tot_rew += env_orig.gamma**cnt * env_orig.rewards[j[0]]
			cnt += 1
		tot_rew_list.append(tot_rew)
		# print(len(traj))
	return tot_rew_list


## data processing file

nx = 24 # columns
ny = 24 # rows
terminal_state = [0]
nx_temp = int(nx/2)
ny_temp = int(ny/2)
gamma = 0.95
noise = 0.1
boltz_beta = 10000

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
(int(np.floor(3*nx/4))+2)*nx+int(np.floor(1*ny/5)),(int(np.floor(5*nx/8))+1)*nx+int(np.floor(5*ny/8))+2,
(int(np.floor(5*nx/8))+1)*nx+int(np.floor(5*ny/8))+1,(int(np.floor(5*nx/8))+2)*nx+int(np.floor(5*ny/8))+1,
(int(np.floor(5*nx/8))+3)*nx+int(np.floor(5*ny/8))+1,(int(np.floor(5*nx/8))+4)*nx+int(np.floor(5*ny/8))+1,
(int(np.floor(5*nx/8))+5)*nx+int(np.floor(5*ny/8))+1,
(int(np.floor(5*nx/8))+5)*nx+int(np.floor(5*ny/8))+2]

constraints[inds] = 1
rewards[0] = 2
rewards[inds] = -10


env_orig = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards, constraints, gamma, noise)

optimal_policy = mdp_utils.get_optimal_policy(env_orig)

# load data

# hierarchical

# 20 demos
mean_cnstr_20_0 = np.loadtxt('plots/mean_constraints_hier_20_0.txt')
mean_cnstr_20_1 = np.loadtxt('plots/mean_constraints_hier_20_1.txt')
mean_cnstr_20_2 = np.loadtxt('plots/mean_constraints_hier_20_2.txt')
mean_cnstr_20_3 = np.loadtxt('plots/mean_constraints_hier_20_3.txt')

map_cnstr_20_0 = np.loadtxt('plots/map_constraints_hier_20_0.txt')
map_cnstr_20_1 = np.loadtxt('plots/map_constraints_hier_20_1.txt')
map_cnstr_20_2 = np.loadtxt('plots/map_constraints_hier_20_2.txt')
map_cnstr_20_3 = np.loadtxt('plots/map_constraints_hier_20_3.txt')

# 50 demos
mean_cnstr_50_0 = np.loadtxt('plots/mean_constraints_hier_50_0.txt')
mean_cnstr_50_1 = np.loadtxt('plots/mean_constraints_hier_50_1.txt')
mean_cnstr_50_2 = np.loadtxt('plots/mean_constraints_hier_50_2.txt')
mean_cnstr_50_3 = np.loadtxt('plots/mean_constraints_hier_50_3.txt')

map_cnstr_50_0 = np.loadtxt('plots/map_constraints_hier_50_0.txt')
map_cnstr_50_1 = np.loadtxt('plots/map_constraints_hier_50_1.txt')
map_cnstr_50_2 = np.loadtxt('plots/map_constraints_hier_50_2.txt')
map_cnstr_50_3 = np.loadtxt('plots/map_constraints_hier_50_3.txt')

# 100 demos
mean_cnstr_100_0 = np.loadtxt('plots/mean_constraints_hier_100_0.txt')
mean_cnstr_100_1 = np.loadtxt('plots/mean_constraints_hier_100_1.txt')
mean_cnstr_100_2 = np.loadtxt('plots/mean_constraints_hier_100_2.txt')
mean_cnstr_100_3 = np.loadtxt('plots/mean_constraints_hier_100_3.txt')

map_cnstr_100_0 = np.loadtxt('plots/map_constraints_hier_100_0.txt')
map_cnstr_100_1 = np.loadtxt('plots/map_constraints_hier_100_1.txt')
map_cnstr_100_2 = np.loadtxt('plots/map_constraints_hier_100_2.txt')
map_cnstr_100_3 = np.loadtxt('plots/map_constraints_hier_100_3.txt')

# 200 demos
mean_cnstr_200_0 = np.loadtxt('plots/mean_constraints_hier_200_0.txt')
mean_cnstr_200_1 = np.loadtxt('plots/mean_constraints_hier_200_1.txt')
mean_cnstr_200_2 = np.loadtxt('plots/mean_constraints_hier_200_2.txt')
mean_cnstr_200_3 = np.loadtxt('plots/mean_constraints_hier_200_3.txt')

map_cnstr_200_0 = np.loadtxt('plots/map_constraints_hier_200_0.txt')
map_cnstr_200_1 = np.loadtxt('plots/map_constraints_hier_200_1.txt')
map_cnstr_200_2 = np.loadtxt('plots/map_constraints_hier_200_2.txt')
map_cnstr_200_3 = np.loadtxt('plots/map_constraints_hier_200_3.txt')

# reshaping data
map_cnstr_20_0 = map_cnstr_20_0.reshape((ny_temp,nx_temp))
map_cnstr_20_1 = map_cnstr_20_1.reshape((ny_temp,nx_temp))
map_cnstr_20_2 = map_cnstr_20_2.reshape((ny_temp,nx_temp))
map_cnstr_20_3 = map_cnstr_20_3.reshape((ny_temp,nx_temp))

map_temp_20 = np.zeros((ny, nx))
map_temp_20[0:ny_temp,0:nx_temp] = map_cnstr_20_0
map_temp_20[0:ny_temp,nx_temp:nx_temp+nx_temp] = map_cnstr_20_1
map_temp_20[ny_temp:ny_temp+ny_temp,0:nx_temp] = map_cnstr_20_2
map_temp_20[ny_temp:ny_temp+ny_temp,nx_temp:nx_temp+nx_temp] = map_cnstr_20_3
map_cnstr_hier_20 = map_temp_20.flatten()

map_cnstr_50_0 = map_cnstr_50_0.reshape((ny_temp,nx_temp))
map_cnstr_50_1 = map_cnstr_50_1.reshape((ny_temp,nx_temp))
map_cnstr_50_2 = map_cnstr_50_2.reshape((ny_temp,nx_temp))
map_cnstr_50_3 = map_cnstr_50_3.reshape((ny_temp,nx_temp))

map_temp_50 = np.zeros((ny, nx))
map_temp_50[0:ny_temp,0:nx_temp] = map_cnstr_50_0
map_temp_50[0:ny_temp,nx_temp:nx_temp+nx_temp] = map_cnstr_50_1
map_temp_50[ny_temp:ny_temp+ny_temp,0:nx_temp] = map_cnstr_50_2
map_temp_50[ny_temp:ny_temp+ny_temp,nx_temp:nx_temp+nx_temp] = map_cnstr_50_3
map_cnstr_hier_50 = map_temp_50.flatten()

map_cnstr_100_0 = map_cnstr_100_0.reshape((ny_temp,nx_temp))
map_cnstr_100_1 = map_cnstr_100_1.reshape((ny_temp,nx_temp))
map_cnstr_100_2 = map_cnstr_100_2.reshape((ny_temp,nx_temp))
map_cnstr_100_3 = map_cnstr_100_3.reshape((ny_temp,nx_temp))

map_temp_100 = np.zeros((ny, nx))
map_temp_100[0:ny_temp,0:nx_temp] = map_cnstr_100_0
map_temp_100[0:ny_temp,nx_temp:nx_temp+nx_temp] = map_cnstr_100_1
map_temp_100[ny_temp:ny_temp+ny_temp,0:nx_temp] = map_cnstr_100_2
map_temp_100[ny_temp:ny_temp+ny_temp,nx_temp:nx_temp+nx_temp] = map_cnstr_100_3
map_cnstr_hier_100 = map_temp_100.flatten()

map_cnstr_200_0 = map_cnstr_200_0.reshape((ny_temp,nx_temp))
map_cnstr_200_1 = map_cnstr_200_1.reshape((ny_temp,nx_temp))
map_cnstr_200_2 = map_cnstr_200_2.reshape((ny_temp,nx_temp))
map_cnstr_200_3 = map_cnstr_200_3.reshape((ny_temp,nx_temp))

map_temp_200 = np.zeros((ny, nx))
map_temp_200[0:ny_temp,0:nx_temp] = map_cnstr_200_0
map_temp_200[0:ny_temp,nx_temp:nx_temp+nx_temp] = map_cnstr_200_1
map_temp_200[ny_temp:ny_temp+ny_temp,0:nx_temp] = map_cnstr_200_2
map_temp_200[ny_temp:ny_temp+ny_temp,nx_temp:nx_temp+nx_temp] = map_cnstr_200_3
map_cnstr_hier_200 = map_temp_200.flatten()



# map_res_20 = [map_cnstr_20_0, map_cnstr_20_1, map_cnstr_20_2, map_cnstr_20_3]
# map_cnstr_hier_20 = np.array([item for sublist in map_res_20 for item in sublist])

# map_res_50 = [map_cnstr_50_0, map_cnstr_50_1, map_cnstr_50_2, map_cnstr_50_3]
# map_cnstr_hier_50 = np.array([item for sublist in map_res_50 for item in sublist])

# map_res_100 = [map_cnstr_100_0, map_cnstr_100_1, map_cnstr_100_2, map_cnstr_100_3]
# map_cnstr_hier_100 = np.array([item for sublist in map_res_100 for item in sublist])

# map_res_200 = [map_cnstr_200_0, map_cnstr_200_1, map_cnstr_200_2, map_cnstr_200_3]
# map_cnstr_hier_200 = np.array([item for sublist in map_res_200 for item in sublist])

# global (single)

map_cnstr_single_20 = np.loadtxt('plots/map_constraints_single_20.txt')
map_cnstr_single_50 = np.loadtxt('plots/map_constraints_single_50.txt')
map_cnstr_single_100 = np.loadtxt('plots/map_constraints_single_100.txt')
map_cnstr_single_200 = np.loadtxt('plots/map_constraints_single_200.txt')
map_cnstr_single_500 = np.loadtxt('plots/map_constraints_single_500.txt')


mean_cnstr_single_20 = np.loadtxt('plots/mean_constraints_single_20.txt')
mean_cnstr_single_50 = np.loadtxt('plots/mean_constraints_single_50.txt')
mean_cnstr_single_100 = np.loadtxt('plots/mean_constraints_single_100.txt')
mean_cnstr_single_200 = np.loadtxt('plots/mean_constraints_single_200.txt')
mean_cnstr_single_500 = np.loadtxt('plots/mean_constraints_single_500.txt')

# classification results


TPR_hier_20, FPR_hier_20, FNR_hier_20 = FP_and_FN_and_TP(constraints, map_cnstr_hier_20)
TPR_hier_50, FPR_hier_50, FNR_hier_50 = FP_and_FN_and_TP(constraints, map_cnstr_hier_50)
TPR_hier_100, FPR_hier_100, FNR_hier_100 = FP_and_FN_and_TP(constraints, map_cnstr_hier_100)
TPR_hier_200, FPR_hier_200, FNR_hier_200 = FP_and_FN_and_TP(constraints, map_cnstr_hier_200)


TPR_single_20, FPR_single_20, FNR_single_20 = FP_and_FN_and_TP(constraints, map_cnstr_single_20)
TPR_single_50, FPR_single_50, FNR_single_50 = FP_and_FN_and_TP(constraints, map_cnstr_single_50)
TPR_single_100, FPR_single_100, FNR_single_100 = FP_and_FN_and_TP(constraints, map_cnstr_single_100)
TPR_single_200, FPR_single_200, FNR_single_200 = FP_and_FN_and_TP(constraints, map_cnstr_single_200)
TPR_single_500, FPR_single_500, FNR_single_500 = FP_and_FN_and_TP(constraints, map_cnstr_single_500)


# hier 20
inds = np.nonzero(map_cnstr_hier_20)
rewards_hier_20 = - np.ones(ny * nx)
rewards_hier_20[0] = 2
rewards_hier_20[inds] = -10

env_hier_20 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards_hier_20, map_cnstr_hier_20, gamma, noise)

# hier 50
inds = np.nonzero(map_cnstr_hier_50)
rewards_hier_50 = - np.ones(ny * nx)
rewards_hier_50[0] = 2
rewards_hier_50[inds] = -10

env_hier_50 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards_hier_50, map_cnstr_hier_50, gamma, noise)

# hier 100
inds = np.nonzero(map_cnstr_hier_100)
rewards_hier_100 = - np.ones(ny * nx)
rewards_hier_100[0] = 2
rewards_hier_100[inds] = -10

env_hier_100 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards_hier_100, map_cnstr_hier_100, gamma, noise)

# hier 200
inds = np.nonzero(map_cnstr_hier_200)
rewards_hier_200 = - np.ones(ny * nx)
rewards_hier_200[0] = 2
rewards_hier_200[inds] = -10

env_hier_200 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards_hier_200, map_cnstr_hier_200, gamma, noise)


# single 20
inds = np.nonzero(map_cnstr_single_20)
rewards_single_20 = - np.ones(ny * nx)
rewards_single_20[0] = 2
rewards_single_20[inds] = -10

env_single_20 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards_single_20, map_cnstr_single_20, gamma, noise)

# single 50
inds = np.nonzero(map_cnstr_single_50)
rewards_single_50 = - np.ones(ny * nx)
rewards_single_50[0] = 2
rewards_single_50[inds] = -10

env_single_50 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards_single_50, map_cnstr_single_50, gamma, noise)

# single 100
inds = np.nonzero(map_cnstr_single_100)
rewards_single_100 = - np.ones(ny * nx)
rewards_single_100[0] = 2
rewards_single_100[inds] = -10

env_single_100 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards_single_100, map_cnstr_single_100, gamma, noise)

# single 200
inds = np.nonzero(map_cnstr_single_200)
rewards_single_200 = - np.ones(ny * nx)
rewards_single_200[0] = 2
rewards_single_200[inds] = -10

env_single_200 = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards_single_200, map_cnstr_single_200, gamma, noise)


# single 500
inds = np.nonzero(map_cnstr_single_500)
rewards_single_500 = - np.ones(ny * nx)
rewards_single_500[0] = 2
rewards_single_500[inds] = -10


plot_grid.plot_grid(nx, ny, env_orig.state_grid_map, 0, 0, constraints=map_cnstr_hier_20)

IPython.embed()

res = {}

res[('hier',20)] = np.mean(eval_policy(env_orig, rewards_hier_20, map_cnstr_hier_20))
res[('hier',50)] = np.mean(eval_policy(env_orig, rewards_hier_50, map_cnstr_hier_50))
res[('hier',100)] = np.mean(eval_policy(env_orig, rewards_hier_100, map_cnstr_hier_100))
res[('hier',200)] = np.mean(eval_policy(env_orig, rewards_hier_200, map_cnstr_hier_200))

res[('single',20)] = np.mean(eval_policy(env_orig, rewards_single_20, map_cnstr_single_20))
res[('single',50)] = np.mean(eval_policy(env_orig, rewards_single_50, map_cnstr_single_50))
res[('single',100)] = np.mean(eval_policy(env_orig, rewards_single_100, map_cnstr_single_100))
res[('single',200)] = np.mean(eval_policy(env_orig, rewards_single_200, map_cnstr_single_200))


IPython.embed()






