import mdp_utils
import mdp_worlds
import bayesian_irl
import copy
import IPython
import plot_grid
import numpy as np
from mdp_utils import calculate_q_values, logsumexp, FP_and_FN_and_TP



## data processing file

nx = 24 # columns
ny = 24 # rows
terminal_state = 0
nx_temp = int(nx/2)
ny_temp = int(ny/2)
gamma = 0.95
noise = 0.1


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

rewards[inds] = -10


env_orig = mdp_worlds.nonrand_gridworld(ny, nx, terminal_state, rewards, constraints, gamma, noise)

optimal_policy = mdp_utils.get_optimal_policy(env2)


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
map_cnstr_hier_100 = map_temp.flatten()

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



map_res_20 = [map_cnstr_20_0, map_cnstr_20_1, map_cnstr_20_2, map_cnstr_20_3]
map_cnstr_hier_20 = np.array([item for sublist in map_res_20 for item in sublist])

map_res_50 = [map_cnstr_50_0, map_cnstr_50_1, map_cnstr_50_2, map_cnstr_50_3]
map_cnstr_hier_50 = np.array([item for sublist in map_res_50 for item in sublist])

map_res_100 = [map_cnstr_100_0, map_cnstr_100_1, map_cnstr_100_2, map_cnstr_100_3]
map_cnstr_hier_100 = np.array([item for sublist in map_res_100 for item in sublist])

map_res_200 = [map_cnstr_200_0, map_cnstr_200_1, map_cnstr_200_2, map_cnstr_200_3]
map_cnstr_hier_200 = np.array([item for sublist in map_res_200 for item in sublist])
IPython.embed()

# global (single)






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

IPython.embed()


optimal_policy = mdp_utils.get_optimal_policy(env2)
