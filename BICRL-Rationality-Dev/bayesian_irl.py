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
        self.state_demos = []
        self.posterior = {new_list: [] for new_list in range(env.num_states)}
        #check to see if FeatureMDP or just plain MDP
        if hasattr(self.env, 'feature_weights'):
            self.num_mcmc_dims = len(self.env.feature_weights)
        else:
            self.num_mcmc_dims = self.env.num_states
        for s, a in self.demonstrations:
            self.state_demos.append(s)
 

    def calc_ll(self, hyp_reward, ind=0):
        #perform hypothetical given current reward hypothesis
        self.env.set_rewards(hyp_reward)
        q_values = calculate_q_values(self.env, epsilon=self.epsilon)
        #calculate the log likelihood of the reward hypothesis given the demonstrations
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        if ind==1:
            IPython.embed()
        for s, a in self.demonstrations:
            # if (s not in self.env.terminals):  # there are no counterfactuals in a terminal state

                Z_exponents = self.beta * q_values[s]
                log_sum += self.beta * q_values[s][a] - logsumexp(Z_exponents)
               
        return log_sum

    def calc_ll_test(self, hyp_reward):
        #perform hypothetical given current reward hypothesis
        self.env.set_rewards(hyp_reward)
        q_values = calculate_q_values(self.env, epsilon=self.epsilon)
        #calculate the log likelihood of the reward hypothesis given the demonstrations
        log_prior = 0.0  #assume unimformative prior
        log_sum = log_prior
        ret_l=[]
        
        for s, a in self.demonstrations:
            # if (s not in self.env.terminals):  # there are no counterfactuals in a terminal state

                Z_exponents = self.beta * q_values[s]
                log_sum += self.beta * q_values[s][a] - logsumexp(Z_exponents)
                ret_l.append(self.beta * q_values[s][a] - logsumexp(Z_exponents))
                if s==0:
                    break
               
        return ret_l



    def generate_proposal_bern_constr(self, old_constr, old_rew_mean, step_size):
        new_constr = copy.deepcopy(old_constr)
        new_rew_mean = copy.deepcopy(old_rew_mean)
        index = np.random.randint(len(old_constr))
        new_constr[index] = 1 if old_constr[index]==0 else 0

        new_rew_mean = new_rew_mean - 1 if np.random.rand() < 0.5 else new_rew_mean + 1
        new_rew = np.random.normal(new_rew_mean, 1)
        
        return new_constr, new_rew, new_rew_mean


    def generate_proposal_bern_constr_alternating(self, old_constr, old_rew, ind, stdev=1):
        new_constr = copy.deepcopy(old_constr)
        new_rew = copy.deepcopy(old_rew)
        if ind % 50 == 0:   
            new_rew = new_rew + stdev * np.random.randn() 
            index = None     
        else:
            index = np.random.randint(len(old_constr))
            new_constr[index] = 1 if old_constr[index]==0 else 0
        # new_rew_mean = new_rew_mean - 1 if np.random.rand() < 0.5 else new_rew_mean + 1
        # new_rew = np.random.normal(new_rew_mean, 1)
        
        return new_constr, new_rew, index

    def initial_solution_bern_cnstr(self):
        # initialize problem solution for MCMC to all zeros, maybe not best initialization but it works in most cases
        new_constr = np.zeros(self.env.num_states)
        new_rew = np.random.randint(-20,-5)
        for i in range(len(new_constr)):
            new_constr[i] = bernoulli.rvs(0.5)
        return new_constr, new_rew

    def compute_variance(self, prob=None, thresh=0.1):
        # compute state with max variance
        vars_list = []
        for i in range(1,self.env.num_states):
            variabs = self.chain_cnstr[:,i]
            theta = np.mean(variabs)
            # print(theta * (1-theta))
            vars_list.append(theta * (1-theta))

        state_query = np.argmax(vars_list) + 1
       
    
        return state_query

 
        
    def run_mcmc_bern_constraint(self, samples, rewards_fix, init_val=None):
        '''
            run metropolis hastings MCMC with Gaussian symmetric proposal and uniform prior
            samples: how many reward functions to sample from posterior
            stepsize: standard deviation for proposal distribution
            normalize: if true then it will normalize the rewards (reward weights) to be unit l2 norm, otherwise the rewards will be unbounded
        '''
        
        num_samples = samples  # number of MCMC samples

        accept_cnt = 0  #keep track of how often MCMC accepts, ideally around 40% of the steps accept
        #if accept count is too high, increase stdev, if too low reduce
        if init_val:
            map_sol = init_val[0]
            map_rew = init_val[1]
        self.chain_cnstr = np.zeros((num_samples, self.num_mcmc_dims)) #store rewards found via BIRL here, preallocate for speed
        self.chain_rew = np.zeros(num_samples)
        # cur_sol = self.initial_solution_bern_cnstr() #initial guess for MCMC
        if init_val:
            cur_constr = init_val[0]
            cur_rew = init_val[1]
        else:
            cur_constr, cur_rew = self.initial_solution_bern_cnstr()
        cur_rew_mean = cur_rew
        cur_sol = copy.deepcopy(rewards_fix)
        for i in range(len(cur_constr)):
            if cur_constr[i] == 1:
                cur_sol[i] = cur_rew
        map_constr = cur_constr
        cur_ll = self.calc_ll(cur_sol)  # log likelihood
        # keep track of MAP loglikelihood and solution
        map_ll = cur_ll  
        map_sol = cur_constr
        map_list = []
        Perf_list = []
        for i in range(num_samples):
          
            prop_constr, prop_rew, index = self.generate_proposal_bern_constr_alternating(cur_constr, cur_rew, i)

            prop_sol = copy.deepcopy(rewards_fix)
            for ii in range(len(prop_constr)):
                if prop_constr[ii] == 1:
                    prop_sol[ii] = prop_rew           

            # calculate likelihood ratio test
            prop_ll = self.calc_ll(prop_sol)

            test = 0
            
            if prop_ll > cur_ll:
                # accept
                self.chain_cnstr[i,np.nonzero(prop_constr)] = 1
                if index != None:
                    self.posterior[index].append(prop_constr[index]) 
                self.chain_rew[i] = prop_rew
                accept_cnt += 1
                cur_constr = prop_constr
                cur_rew = prop_rew
                # cur_rew_mean = prop_rew_mean
                cur_sol = prop_sol
                cur_ll = prop_ll
                if prop_ll > map_ll:  # maxiumum aposterioi
                    map_ll = prop_ll
                    map_constr = prop_constr
                    map_rew = prop_rew
                    map_sol = prop_constr
                    map_list.append(prop_constr)
            else:
                # accept with prob exp(prop_ll - cur_ll)
                if np.random.rand() < np.exp(prop_ll - cur_ll):
                    self.chain_cnstr[i,np.nonzero(prop_constr)] = 1
                    self.chain_rew[i] = prop_rew
                    if index != None:
                        self.posterior[index].append(prop_constr[index]) 
                    accept_cnt += 1
                    cur_sol = prop_sol
                    cur_constr = prop_constr
                    cur_rew = prop_rew
                    cur_ll = prop_ll

                else:
                    # reject
                    self.chain_cnstr[i,np.nonzero(cur_constr)] = 1
                    if index != None:
                        self.posterior[index].append(1-prop_constr[index]) 
                    self.chain_rew[i] = cur_rew
            # if i ==200 or i==num_samples-1:
            #     plot_grid.plot_grid(8, 6, self.env.state_grid_map, i, map_sol)
            #     plot_grid_mean_constr(nx, ny, env2.state_grid_map, kk, constraints, mean_constraints, trajectory_demos, optimal_policy)
            TPR, FPR, FNR, Precision = FP_and_FN_and_TP(self.env.constraints, map_constr)
            # print(i, TPR, FPR, FNR, Precision)
            Perf_list.append((i,TPR,FPR,FNR,Precision))
        # print("accept rate:", accept_cnt / num_samples)
        self.accept_rate = accept_cnt / num_samples
        self.map_sol = map_sol
        self.map_rew = map_rew
        self.map_list = map_list
        self.Perf_list = Perf_list

        

    def get_map_solution(self):
        return self.map_sol, self.map_rew


    def get_mean_solution(self, burn_frac=0.1, skip_rate=1):
        ''' get mean solution after removeing burn_frac fraction of the initial samples and only return every skip_rate
            sample. Skiping reduces the size of the posterior and can reduce autocorrelation. Burning the first X% samples is
            often good since the starting solution for mcmc may not be good and it can take a while to reach a good mixing point
        '''

        burn_indx = int(len(self.chain_cnstr) * burn_frac)
        # IPython.embed()
        mean_cnstr = np.mean(self.chain_cnstr[burn_indx::skip_rate], axis=0)

        burn_indx = int(len(self.chain_rew) * burn_frac)
        
        mean_rew = np.mean(self.chain_rew[burn_indx::skip_rate], axis=0)
        
        return mean_cnstr, mean_rew

