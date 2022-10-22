import mdp_utils
import mdp_worlds
import bayesian_irl_rew, bayesian_bicrl, bayesian_irl_rew_continuous, bayesian_irl_rew_discrete
import copy
import IPython
import plot_grid
import numpy as np
import pickle, argparse





parser = argparse.ArgumentParser()
parser.add_argument('--nx', type=int, default=3)
parser.add_argument('--ny', type=int, default=11)
parser.add_argument('--noise', type=float, default=0.1)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--boltz_beta', type=float, default=1)
parser.add_argument('--stdev', type=float, default=0.1)
parser.add_argument('--num_steps', type=int, default=4000)
parser.add_argument('--iterations', type=int, default=1)
parser.add_argument('--N_demonstrations', type=int, default=100)
parser.add_argument('--method', type=str, default='bicrl')
args = parser.parse_args()

def main():

    nx = args.nx # columns
    ny = args.ny # rows
    gamma = args.gamma
    noise = args.noise
    boltz_beta = args.boltz_beta
    num_steps = args.num_steps
    iterations = args.iterations
    stdev = args.stdev
    N_demonstrations = args.N_demonstrations
    method = args.method
    res_dict = {}

    print("create a random ny x nx gridworld with no featurized reward function")
    # rewards negative everywhere except for the goal state
    terminal_states = [0]#[rows * columns - 1]
    # rewards negative everywhere except for the goal state
    rewards = - np.zeros(ny * nx)
    
    # constraints is a binary vector indicating hte presence or not of a constraint
    constraints = np.zeros(ny * nx) # not used

    rewards_fix = copy.deepcopy(rewards)
    env = mdp_worlds.highway(ny, nx, terminal_states, rewards, constraints, gamma, noise)
    # actual feature weights
    W_real = np.zeros(env.num_features)
    W_real[0:3] = -10.0
    W_real[3] = 1.0
    W_real[4:7] = -1
    W_real[7] = -10.0
    W_real[8] = 10.0

    W_fix = np.zeros(env.num_features)
    W_fix[4:7] = W_real[4:7]
    W_fix[3] = W_real[3]
    W_fix[8] = W_real[8]

    real_rewards = env.get_rewards(W_real, None)
    env.set_rewards(real_rewards)
    mean_penalty_list = []
    res_dict = {}

    for iteration in range(iterations):
        trajectory_demos = []
  
        env_orig = copy.deepcopy(env)
        perf_lists = []
        trajectory_demos = []
        start_state = env.num_states - 2
        # obtain trajectories
        for i in range(N_demonstrations):
            trajectory_demos.append(mdp_utils.generate_boltzman_demo(env, boltz_beta, start_state))
       
        trajectory_demos = [item for sublist in trajectory_demos for item in sublist]
       

        print('Method {}, Iteration {}'.format(method, iteration))
               
        if method == 'pen_rew_disc':
            birl = bayesian_irl_rew_discrete.BIRL(env, trajectory_demos, boltz_beta)
            birl.run_mcmc_bern_constraint(num_steps, W_fix)
            chain = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
            acc_rate = birl.accept_rate
            print('acceptance rate {}'.format(acc_rate))
            map_rew = birl.get_map_solution()
           
            res_dict[iteration,'map'] = map_rew
            res_dict[iteration,'chain'] = chain

        elif method == 'pen_rew_cont': 
            birl = bayesian_irl_rew_continuous.BIRL(env, trajectory_demos, boltz_beta)
            birl.run_mcmc_bern_constraint(num_steps, W_fix, stdev)
            chain = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
            acc_rate = birl.accept_rate
            print('acceptance rate {}'.format(acc_rate))
            map_rew = birl.get_map_solution()
            
            res_dict[iteration,'map'] = map_rew
            res_dict[iteration,'chain'] = chain

        elif method == 'BIRL': 
            birl = bayesian_irl_rew.BIRL(env, trajectory_demos, boltz_beta)
            birl.run_mcmc_bern_constraint(num_steps, rewards_fix, stdev)
            chain = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
            acc_rate = birl.accept_rate
            print('acceptance rate {}'.format(acc_rate))
            map_rew = birl.get_map_solution()
           
            res_dict[iteration,'chain'] = chain
            res_dict[iteration,'map'] = map_rew
            
        elif method == 'BICRL':
            birl = bayesian_bicrl.BIRL(env, trajectory_demos, boltz_beta)
            birl.run_mcmc_bern_constraint(num_steps, W_fix, stdev)
            acc_rate = birl.accept_rate
            print('acceptance rate {}'.format(acc_rate)) 
            chain_constraints = birl.get_mean_solution(burn_frac=0.2, skip_rate=1)
            map_constraints, map_rew = birl.get_map_solution()
            res_dict[iteration,'chain'] = chain_constraints
            res_dict[iteration,'map'] = [map_constraints, map_rew]
            

        # np.savetxt('plots/feat_mean_constraints_single_' + str(kk) + '.txt', mean_constraints)
        # # np.savetxt('plots/map_constraints_single_' + str(N_demos)+'_'+str(kk) + '.txt', map_constraints)
    filename = 'data/' + method + '_' + str(num_steps) + '_' + str(N_demonstrations)
    with open(filename + '.pickle', 'wb') as handle:
        pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    main()

