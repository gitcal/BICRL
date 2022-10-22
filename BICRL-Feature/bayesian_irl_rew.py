from mdp_utils import calculate_q_values, logsumexp, FP_and_FN_and_TP
import numpy as np
import copy
from random import choice
import IPython
from scipy.stats import bernoulli
import plot_grid

class BIRL:
    def __init__(self, env, demos, beta, num_cnstr=0, epsilon=0.001):

        """
        Class for running and storing output of mcmc for Bayesian IRL
        env: the mdp (we ignore the reward)
        demos: list of (s,a) tuples 
        beta: the assumed boltzman rationality of the demonstrator

        """
        self.env = copy.deepcopy(env)
        self.demonstrations = demos
        self.epsilon = epsilon
        self.beta = beta
        self.num_cnstr = num_cnstr
        # self.posterior = {new_list: [] for new_list in range(env.num_states)}
        #check to see if FeatureMDP or just plain MDP
        
        self.num_mcmc_dims = self.env.num_features
        # np.random.seed(10)

    
    def calc_ll(self, cur_sol):
        #perform hypothetical given current reward hypothesis
        self.env.set_rewards(cur_sol)
        q_values = calculate_q_values(self.env, epsilon=self.epsilon)
        #calculate the log likelihood of the reward hypothesis given the demonstrations
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        for s, a in self.demonstrations:
            if (s not in self.env.terminals):  # there are no counterfactuals in a terminal state
                Z_exponents = self.beta * q_values[s]
                log_sum += self.beta * q_values[s][a] - logsumexp(Z_exponents)         
        return log_sum



    def generate_proposal_bern_constr(self, W_old):
        W_new = copy.deepcopy(W_old)
        index = np.random.randint(len(W_old))
        W_new[index] = np.random.normal(W_old[index], 1)

        return W_new



    def generate_proposal_bern_constr_alternating(self, W_old, stdev=0.2):
        W_new = copy.deepcopy(W_old)
                
        index = np.random.randint(len(W_old))
        # if W_old[index] == 0:
        temp = W_old[index] + stdev * np.random.randn() 
        W_new[index] = temp
       
        
        return W_new
                


    def initial_solution_bern_cnstr(self):
        # initialize problem solution for MCMC to all zeros, maybe not best initialization but it works in most cases
        # W_new = np.random.choice([0, 1], size=(self.env.num_features,))
        W_new = np.random.randn(self.env.num_features)
        pen_rew = np.random.randint(-15, -1)#, self.env.num_states)
        W_new = np.zeros(self.env.num_features)
        return W_new

  

    def run_mcmc_bern_constraint(self, samples, W_fix, stdev=1):
        '''
            run metropolis hastings MCMC with Gaussian symmetric proposal and uniform prior
            samples: how many reward functions to sample from posterior
            stepsize: standard deviation for proposal distribution
            normalize: if true then it will normalize the rewards (reward weights) to be unit l2 norm, otherwise the rewards will be unbounded
        '''
        
        num_samples = samples  # number of MCMC samples
        # stdev = stepsize  # initial guess for standard deviation, doesn't matter too much

        accept_cnt = 0  #keep track of how often MCMC accepts, ideally around 40% of the steps accept
        #if accept count is too high, increase stdev, if too low reduce

        self.chain_W = np.zeros((num_samples, self.num_mcmc_dims)) #store rewards found via BIRL here, preallocate for speed
        cur_W = self.initial_solution_bern_cnstr()     
        cur_sol = self.env.get_rewards(cur_W, None)
       
        # IPython.embed()
        map_W = cur_W
        cur_ll = self.calc_ll(cur_sol)  # log likelihood
        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll  
        map_sol = cur_sol
        map_list = []
        Perf_list = []
        for i in range(num_samples):
            # sample from proposal distribution
            # print(i)
            
            prop_W = self.generate_proposal_bern_constr_alternating(cur_W, stdev)
            prop_sol = self.env.get_rewards(prop_W, None)
            # calculate likelihood ratio test
            prop_ll = self.calc_ll(prop_sol)
           
            # print('prop',prop_W, prop_rew)
            if prop_ll > cur_ll:
                # print('accept')
                # accept
                # IPython.embed()
                self.chain_W[i,:] = prop_W               
             
                accept_cnt += 1
                cur_W = prop_W
              
                cur_ll = prop_ll
                if prop_ll > map_ll:  # maxiumum aposterioi
                    map_ll = prop_ll
                    map_W = prop_W
                    map_sol = prop_W
                    # map_list.append(prop_W)
            else:
                # accept with prob exp(prop_ll - cur_ll)
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                    self.chain_W[i,:] = prop_W
                    # self.chain_rew[i,:] = prop_rew
                    accept_cnt += 1
                    # cur_sol = prop_sol
                    cur_W = prop_W
                    # cur_rew = prop_rew
                    cur_ll = prop_ll
                    # cur_rew_mean = prop_rew_mean

                else:
                    # reject
                    self.chain_W[i,:] = cur_W
                    
                    # self.chain_rew[i,:] = cur_rew
            
            # if i ==200 or i==num_samples-1:
            #     plot_grid.plot_grid(8, 6, self.env.state_grid_map, i, map_sol)
            #     plot_grid_mean_constr(nx, ny, env2.state_grid_map, kk, constraints, mean_constraints, trajectory_demos, optimal_policy)
            # TPR, FPR, FNR = FP_and_FN_and_TP(self.env_orig.constraints, map_constr)
            # Perf_list.append((i,TPR,FPR,FNR))
        print("accept rate:", accept_cnt / num_samples)
        self.accept_rate = accept_cnt / num_samples
        self.map_sol = map_sol
        self.map_list = map_list
        self.Perf_list = Perf_list
        # print("MAP Loglikelihood", map_ll)
        # print("MAP reward")
        # print_array_as_grid(map_sol, mdp)
        # print(cur_rew)
         

    def get_map_solution(self):
        return self.map_sol


    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        ''' get mean solution after removeing burn_frac fraction of the initial samples and only return every skip_rate
            sample. Skiping reduces the size of the posterior and can reduce autocorrelation. Burning the first X% samples is
            often good since the starting solution for mcmc may not be good and it can take a while to reach a good mixing point
        '''

        # burn_indx = int(len(self.chain_cnstr) * burn_frac)
        # mean_cnstr = np.mean(self.chain_cnstr[burn_indx::skip_rate], axis=0)

        # burn_indx = int(len(self.chain_rew) * burn_frac)
        Chain_W = self.chain_W#[burn_indx::skip_rate]
        # np.mean(self.chain_cnstr[burn_indx::skip_rate], axis=0)
        
        return Chain_W
