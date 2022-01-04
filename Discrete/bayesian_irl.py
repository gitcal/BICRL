from mdp_utils import calculate_q_values, logsumexp, FP_and_FN_and_TP
import numpy as np
import copy
from random import choice
import IPython
from scipy.stats import bernoulli
import plot_grid

class BIRL:
    def __init__(self, env, demos, beta, env_orig, num_cnstr=0, epsilon=0.0001):

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
        self.env_orig = env_orig
        self.posterior = {new_list: [] for new_list in range(env.num_states)}
        #check to see if FeatureMDP or just plain MDP
        if hasattr(self.env, 'feature_weights'):
            self.num_mcmc_dims = len(self.env.feature_weights)
        else:
            self.num_mcmc_dims = self.env.num_states
        np.random.seed(10)

        

    

    def calc_ll(self, hyp_reward):
        #perform hypothetical given current reward hypothesis
        # IPython.embed()
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



    def generate_proposal_bern_constr(self, old_constr, old_rew_mean, step_size):
        new_constr = copy.deepcopy(old_constr)
        new_rew_mean = copy.deepcopy(old_rew_mean)
        index = np.random.randint(len(old_constr))
        new_constr[index] = 1 if old_constr[index]==0 else 0

        new_rew_mean = new_rew_mean - 1 if np.random.rand() < 0.5 else new_rew_mean + 1
        new_rew = np.random.normal(new_rew_mean, 1)
        
        return new_constr, new_rew, new_rew_mean


    def generate_proposal_bern_constr_alternating(self, old_constr, old_rew, step_size, ind = 0, stdev = 1):
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


    def compute_variance(self, prob=None, thresh=0.1):

        vars_list = []
        for i in range(1,self.env.num_states):
            variabs = self.chain_cnstr[:,i]
            theta = np.mean(variabs)
            # print(theta * (1-theta))
            vars_list.append(theta * (1-theta))

        # cnt = 1
        # high_var_ind_list = []
        # for val in vars_list:
        #     if val > thresh:
        #         high_var_ind_list.append(cnt)
        #     cnt += 1

        # IPython.embed()
        # ind = [i for i in vars_list if i > thresh]# and i not in self.env_orig.constriants]
        
        # high_var_ind_list = [i for i, x in enumerate(ind) if x] 
        # temp_dems = np.array(self.demonstrations)
        # info = []
        # for i in high_var_ind_list:
        #     temp_state = np.expand_dims(np.array(self.env.state_grid_map[i]),0)
            
            
        #     temp_dists = cdist(temp_state, temp_dems, 'cityblock')
        #     temp_ind = np.argmin(temp_dists)
        #     min_dist = 1/(np.min(temp_dists)+0.01)
        #     min_dist = (min_dist-0)/(0.25-0)
        #     info.append((self.env.state_grid_map[i],min_dist + vars_list[i]))

        #%%%
        state_query = np.argmax(vars_list) + 1
        # state_query = self.env.state_grid_map[temp_listtt]

        # IPython.embed()
        # #%%%

        # temp_list = sorted(info, key=lambda x: x[1])
        # # IPython.embed()
        # if temp_list:
        #     state_query = temp_list[-1][0]
        # else:
        #     state_query = None

        
            # dists = np.kinald.norm()

    
        return state_query
                


    def initial_solution_bern_cnstr(self):
        # initialize problem solution for MCMC to all zeros, maybe not best initialization but it works in most cases
        new_constr = np.zeros(self.env.num_states)
        new_rew = np.random.randint(-20,-5)
        for i in range(len(new_constr)):
            new_constr[i] = bernoulli.rvs(0.5)
        return new_constr, new_rew

  
        
    def run_mcmc_bern_constraint(self, samples, stepsize, N_demos, rewards_fix, normalize=True):
        '''
            run metropolis hastings MCMC with Gaussian symmetric proposal and uniform prior
            samples: how many reward functions to sample from posterior
            stepsize: standard deviation for proposal distribution
            normalize: if true then it will normalize the rewards (reward weights) to be unit l2 norm, otherwise the rewards will be unbounded
        '''
        
        num_samples = samples  # number of MCMC samples
        stdev = stepsize  # initial guess for standard deviation, doesn't matter too much

        accept_cnt = 0  #keep track of how often MCMC accepts, ideally around 40% of the steps accept
        #if accept count is too high, increase stdev, if too low reduce

        self.chain_cnstr = np.zeros((num_samples, self.num_mcmc_dims)) #store rewards found via BIRL here, preallocate for speed
        self.chain_rew = np.zeros(num_samples)
        # cur_sol = self.initial_solution_bern_cnstr() #initial guess for MCMC
        cur_constr, cur_rew = self.initial_solution_bern_cnstr()
        cur_rew_mean = cur_rew
        # print(cur_prob)
        cur_sol = copy.deepcopy(rewards_fix)
        for i in range(len(cur_constr)):
                if cur_constr[i] == 1:
                    cur_sol[i] = cur_rew



        
        map_constr = cur_constr
        cur_ll = self.calc_ll(cur_sol)  # log likelihood
        #keep track of MAP loglikelihood and solution
        map_ll = cur_ll  
        map_sol = cur_sol
        map_list = []
        Perf_list = []
        for i in range(num_samples):
            # sample from proposal distribution
            # prop_constr, prop_rew, prop_rew_sample = self.generate_proposal_bern_constr2(cur_constr, cur_rew, stepsize, normalize, i)
            prop_constr, prop_rew, index = self.generate_proposal_bern_constr_alternating(cur_constr, cur_rew, stepsize, i, 1)
            # prop_constr, prop_rew, prop_rew_mean = self.generate_proposal_bern_constr(cur_constr, cur_rew_mean, stepsize)

            # IPython.embed()
            prop_sol = copy.deepcopy(rewards_fix)
            for ii in range(len(prop_constr)):
                if prop_constr[ii] == 1:
                    prop_sol[ii] = prop_rew

            # IPython.embed()
            # print(prop_sol)
            # calculate likelihood ratio test
            prop_ll = self.calc_ll(prop_sol)
            print(i,cur_ll,prop_ll)
            if prop_ll > cur_ll:
                # accept
                # IPython.embed()
                self.chain_cnstr[i,np.nonzero(prop_constr)] = 1
                if index != None:
                    self.posterior[index].append(prop_constr[index]) 
                self.chain_rew[i] = prop_rew
                accept_cnt += 1
                cur_constr = prop_constr
                cur_rew = prop_rew
                # cur_rew_mean = prop_rew_mean
                # cur_sol = prop_sol
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
                    # cur_sol = prop_sol
                    cur_constr = prop_constr
                    cur_rew = prop_rew
                    cur_ll = prop_ll
                    # cur_rew_mean = prop_rew_mean

                else:
                    # reject
                    self.chain_cnstr[i,np.nonzero(cur_constr)] = 1
                    if index != None:
                        self.posterior[index].append(1-prop_constr[index]) 
                    self.chain_rew[i] = cur_rew
            if i%501==0 or i==num_samples-1:
                # plot_grid.plot_grid(8, 6, self.env.state_grid_map, i, map_sol)
                # IPython.embed()
                # mean_constraints = np.mean(self.chain_cnstr[i-100:i,:],axis=0)
                mean_constraints = np.mean(self.chain_cnstr[i - int(np.floor((3/4*i))):i,:],axis=0)
                name = 'single_mean_cnstr_' + str(N_demos) + '_'
                plot_grid.plot_grid_mean_constr(self.env.num_rows, self.env.num_rows, self.env.state_grid_map, i, name, self.env.constraints, mean_constraints)#, trajectory_demos, optimal_policy)
            # TPR, FPR, FNR = FP_and_FN_and_TP(self.env_orig.constraints, map_constr)
            # Perf_list.append((i,TPR,FPR,FNR))

        print("accept rate:", accept_cnt / num_samples)
        self.accept_rate = accept_cnt / num_samples
        self.map_sol = map_sol
        self.map_rew = map_rew
        self.map_list = map_list
        self.Perf_list = Perf_list
        # print("MAP Loglikelihood", map_ll)
        # print("MAP reward")
        # print_array_as_grid(map_sol, mdp)
        print(cur_rew)
        # IPython.embed()
        

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

