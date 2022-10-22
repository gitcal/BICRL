from mdp_utils import calculate_q_values, logsumexp, FP_and_FN_and_TP
import numpy as np
import copy
from random import choice
import IPython
from scipy.stats import bernoulli
import plot_grid
from numpy.random import choice


class BIRL:
    def __init__(self, env, demos, beta, num_cnstr=0, epsilon=0.0001):

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
        self.posterior = {new_list: [] for new_list in range(env.num_states)}
        #check to see if FeatureMDP or just plain MDP
        self.num_mcmc_dims = self.env.num_features
 

    def calc_ll(self, hyp_reward):
        #perform hypothetical given current reward hypothesis
        self.env.set_rewards(hyp_reward)
        q_values = calculate_q_values(self.env, epsilon=self.epsilon)
        #calculate the log likelihood of the reward hypothesis given the demonstrations
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        for s, a in self.demonstrations:
            if (s not in self.env.terminals):  # there are no counterfactuals in a terminal state
                Z_exponents = self.beta * q_values[s]
                log_sum += self.beta * q_values[s][a] - logsumexp(Z_exponents)
               
        return log_sum


    def generate_proposal(self, old_rew):
        new_rew = copy.deepcopy(old_rew)
        # for continuous uncomment following two lines and comment out for loop
        # index = np.random.randint(len(old_rew))
        # new_rew[index] = old_rew[index] + 1 if np.random.rand() < 0.5  else  old_rew[index] - 1
        for index in range(len(old_rew)):
            new_rew[index] = old_rew[index] + 1 if np.random.rand() < 0.5  else  old_rew[index] - 1        
        return new_rew

    def initial_proposal(self):
        # initialize problem solution for MCMC to all zeros, maybe not best initialization but it works in most cases
        new_W = np.random.randint(-15,-1, size=self.env.num_states)
        return new_W

    def generate_proposal(self, W_old, stdev=0.1):

        W_new = copy.deepcopy(W_old)   
        index = np.random.randint(len(W_old))
        # IPython.embed()
        W_new[index] =  W_old[index] + 1 if np.random.rand() < 0.5  else  W_old[index] - 1    
        
        return W_new

    def initial_proposal(self):
        # initialize problem solution for MCMC to all zeros, maybe not best initialization but it works in most cases
        # W_new = np.random.choice([0, 1], size=(self.env.num_features,))
        W_new = np.random.randint(-15,-1, size=self.num_mcmc_dims)#np.zeros(self.env.num_features)
        W_new = np.zeros(self.num_mcmc_dims)#np.zeros(self.env.num_features)
        # pen_rew = np.random.randint(-10, -1)#, self.env.num_states)

        return W_new
        
    def run_mcmc_bern_constraint(self, samples, W_fix):
        '''
            run metropolis hastings MCMC with Gaussian symmetric proposal and uniform prior
            samples: how many reward functions to sample from posterior
            stepsize: standard deviation for proposal distribution
            normalize: if true then it will normalize the rewards (reward weights) to be unit l2 norm, otherwise the rewards will be unbounded
        '''
        
        num_samples = samples  # number of MCMC samples
        accept_cnt = 0  #keep track of how often MCMC accepts, ideally around 40% of the steps accept
        #if accept count is too high, increase stdev, if too low reduce
        self.chain_W = np.zeros((num_samples, self.num_mcmc_dims)) #store rewards found via BIRL here, preallocate for speed
        cur_W_add = self.initial_proposal()
        cur_W = W_fix + cur_W_add  
        cur_sol = self.env.get_rewards(cur_W_add, None)
 
        cur_ll = self.calc_ll(cur_sol)  # log likelihood
        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll  
        map_sol = cur_sol
        map_W = cur_W_add
        for i in range(num_samples):
            # sample from proposal distribution
            prop_W_add = self.generate_proposal(cur_W_add)
            prop_W = W_fix + prop_W_add
            prop_sol = self.env.get_rewards(prop_W, None)
            # calculate likelihood ratio test
            prop_ll = self.calc_ll(prop_sol)
            if prop_ll > cur_ll:
                # accept
                self.chain_W[i,:] = prop_W
                accept_cnt += 1
                cur_W_add = prop_W_add
                cur_sol = prop_sol
                cur_ll = prop_ll
                if prop_ll > map_ll:  # maxiumum aposterioi
                    map_ll = prop_ll
                    map_W = prop_W
                    map_sol = prop_sol
            else:
                # accept with prob exp(prop_ll - cur_ll)
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                    self.chain_W[i,:] = prop_W
                    accept_cnt += 1
                    cur_sol = prop_sol
                    cur_W_add = prop_W_add
                    cur_ll = prop_ll
                else:
                    # reject
                    self.chain_W[i,:] = cur_W
        

        # print("accept rate:", accept_cnt / num_samples)
        self.accept_rate = accept_cnt / num_samples
        self.map_sol = map_sol
        self.map_W = map_W
        
        

    def get_map_solution(self):
        
        return self.map_W


    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        ''' get mean solution after removeing burn_frac fraction of the initial samples and only return every skip_rate
            sample. Skiping reduces the size of the posterior and can reduce autocorrelation. Burning the first X% samples is
            often good since the starting solution for mcmc may not be good and it can take a while to reach a good mixing point
        '''
        # burn_indx = int(len(self.chain_rew) * burn_frac)        
        mean_W = np.mean(self.chain_W, axis=0)
        
        return self.chain_W#[burn_indx::skip_rate]

