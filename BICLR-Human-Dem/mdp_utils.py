# from mdp import MDP, FeatureMDP
from matplotlib import pyplot as plt
import time
import numpy as np
import math
import copy
import IPython
import cont_env_2d

def value_iteration(env, epsilon=0.001):
    """
    TODO: speed up 
  :param env: the MDP
  :param epsilon: numerical precision for values
  :return:
  """
    n = env.num_states
    # V = np.zeros(n)  # could also use np.zero(n)
    V = -100 * np.ones(n)
    Delta = np.inf #something large to make sure we enter while loop
    # epsilon=1
    while Delta > epsilon * (1 - env.gamma) / env.gamma:
        V_old = V.copy()
        Delta = 0

        for s in range(n):
            max_action_value = -math.inf

            for a in range(env.num_actions):
                action_value = np.dot(env.transitions[s][a], V_old)
                max_action_value = max(action_value, max_action_value)
            V[s] = env.rewards[s] + env.gamma * max_action_value
            # print(s,V[s])
            if abs(V[s] - V_old[s]) > Delta:
                Delta = abs(V[s] - V_old[s])
        # print(Delta)
    return V


def policy_evaluation(policy, env, epsilon):
    """
  Evalute the policy and compute values in each state when executing the policy in the mdp
  :param policy: the policy to evaluate in the mdp
  :param env: markov decision process where we evaluate the policy
  :param epsilon: numerical precision desired
  :return: values of policy under mdp
  """
    n = env.num_states
    V = np.zeros(n)  # could also use np.zero(n)
    Delta = 10
    
    while Delta > epsilon:
        V_old = V.copy()
        Delta = 0
        for s in range(n):
            a = policy[s]
            policy_action_value = np.dot(env.transitions[s][a], V_old)
            V[s] = env.rewards[s] + env.gamma * policy_action_value
            if abs(V[s] - V_old[s]) > Delta:
                Delta = abs(V[s] - V_old[s])

    return V


def get_optimal_policy(env, epsilon=0.0001, V=None):
    #runs value iteration if not supplied as input
    if not V:
        V = value_iteration(env, epsilon)
    
    n = env.num_states
    optimal_policy = []  # our game plan where we need to

    for s in range(n):
        max_action_value = -math.inf
        best_action = 0

        for a in range(env.num_actions):
            action_value = 0.0
            for s2 in range(n):  # look at all possible next states
                
                action_value += env.transitions[s][a][s2] * V[s2]
                # check if a is max
            if action_value > max_action_value:
                max_action_value = action_value
                best_action = a  # direction to take
        optimal_policy.append(best_action)
    return optimal_policy


def logsumexp(x):
    max_x = np.max(x)
    sum_exp = 0.0
    for xi in x:
        sum_exp += np.exp(xi - max_x)
    return max(x) + np.log(sum_exp)


def sumexp(x, beta):
    # max_x = np.max(x)
    sum_exp = 0.0
    for xi in x:
        sum_exp += np.exp(beta*xi)
    return sum_exp


def demonstrate_entire_optimal_policy(env):
    opt_pi = get_optimal_policy(env)
    demo = []

    for state, action in enumerate(opt_pi):
        demo.append((state, action))

    return demo



def calculate_q_values(env, V=None, epsilon=0.0001):
    """
  gets q values for a markov decision process

  :param env: markov decision process
  :param epsilon: numerical precision
  :return: reurn the q values which are
  """

    #runs value iteration if not supplied as input
    if type(V) == type(None):
        V = value_iteration(env, epsilon)
    n = env.num_states

    Q_values = -100 * np.ones((n, env.num_actions)) # was np.zeros
    for s in range(n):
        # if s not in np.nonzero(env.constraints)[0]:
        for a in range(env.num_actions):
            Q_values[s][a] = env.rewards[s] + env.gamma * np.dot(env.transitions[s][a], V)
    
    return Q_values




def action_to_string(act, UP=0, DOWN=1, LEFT=2, RIGHT=3):
    if act == UP:
        return "^"
    elif act == DOWN:
        return "v"
    elif act == LEFT:
        return "<"
    elif act == RIGHT:
        return ">"
    else:
        return NotImplementedError


def visualize_trajectory(trajectory, env):
    """input: list of (s,a) tuples and mdp env
        ouput: prints to terminal string representation of trajectory"""
    states, actions = zip(*trajectory)
    count = 0
    for r in range(env.num_rows):
        policy_row = ""
        for c in range(env.num_cols):
            if count in states:
                #get index
                indx = states.index(count)
                if count in env.terminals:
                    policy_row += ".\t"    
                else:    
                    policy_row += action_to_string(actions[indx]) + "\t"
            else:
                policy_row += " \t"
            count += 1
        print(policy_row)



def visualize_policy(policy, env):
    """
  prints the policy of the MDP using text arrows and uses a '.' for terminals
  """
    count = 0
    for r in range(env.num_rows):
        policy_row = ""
        for c in range(env.num_cols):
            if count in env.terminals:
                policy_row += ".\t"    
            else:
                policy_row += action_to_string(policy[count]) + "\t"
            count += 1
        print(policy_row)


def print_array_as_grid(array_values, env):
    """
  Prints array as a grid
  :param array_values:
  :param env:
  :return:
  """
    count = 0
    for r in range(env.num_rows):
        print_row = ""
        for c in range(env.num_cols):
            print_row += "{:.2f}\t".format(array_values[count])
            count += 1
        print(print_row)


def arg_max_set(values, eps=0.0001):
    # return a set of the indices that correspond to the maximum element(s) in the set of values
    # input is a list or 1-d array and eps tolerance for determining equality
    max_val = max(values)
    arg_maxes = []  # list for storing the indices to the max value(s)
    for i, v in enumerate(values):
        if abs(max_val - v) < eps:
            arg_maxes.append(i)
    return arg_maxes


def calculate_percentage_optimal_actions(pi, env, epsilon=0.0001):
    # calculate how many actions under pi are optimal under the env
    accuracy = 0.0
    # first calculate the optimal q-values under env
    q_values = calculate_q_values(env, epsilon=epsilon)
    # then check if the actions under pi are maximizing the q-values
    for state, action in enumerate(pi):
        if action in arg_max_set(q_values[state], epsilon):
            accuracy += 1  # policy action is an optimal action under env

    return accuracy / env.num_states


def calculate_expected_value_difference(eval_policy, env, epsilon=0.0001):
    '''calculates the difference in expected returns between an optimal policy for an mdp and the eval_policy'''
    V_opt = value_iteration(env, epsilon)
    V_eval = policy_evaluation(eval_policy, env, epsilon)

    return np.mean(V_opt) - np.mean(V_eval)


def generate_optimal_demo(env, start_state):
    """
    Genarates a single optimal demonstration consisting of state action pairs(s,a)
    :param env: Markov decision process passed by main see (markov_decision_process.py)
    :param beta: Beta is a rationality quantification
    :param start_state: start state of demonstration
    :return:
    """
    current_state = start_state
    max_traj_length = env.num_states  #this should be sufficiently long, maybe too long...
    optimal_trajectory = []
    q_values = calculate_q_values(env)

    while (
        current_state not in env.terminals  #stop when we reach a terminal
        and len(optimal_trajectory) < max_traj_length
    ):  # need to add a trajectory length for infinite mdps
        #generate an optimal action, break ties uniformly at random
        act = np.random.choice(arg_max_set(q_values[current_state]))
        optimal_trajectory.append((current_state, act))
        probs = env.transitions[current_state][act]
        probs = probs/np.sum(probs)# cheating
        next_state = np.random.choice(env.num_states, p=probs)
        current_state = next_state
        # this if statement makes sure the temrinal state is included in the trajectory
        if current_state in env.terminals:
            optimal_trajectory.append((current_state, act))

    return optimal_trajectory


def generate_boltzman_demo(env, V, beta, start_state):
    """
    Genarates a single boltzman rational demonstration consisting of state action pairs(s,a)
    :param env: Markov decision process passed by main see (markov_decision_process.py)
    :param beta: Beta is a rationality quantification
    :param start_state: start state of demonstration
    :return:
    """
    current_state = start_state
    max_traj = env.num_states // 2  #this should be sufficiently long, maybe too long...
    boltzman_rational_trajectory = []
    q_values = calculate_q_values(env, V)

    while (
        current_state not in env.terminals  #stop when we reach a terminal
        and len(boltzman_rational_trajectory) < max_traj
    ):  # need to add a trajectory length for infinite envs

        log_numerators = beta * np.array(q_values[current_state])
        boltzman_log_probs = log_numerators - logsumexp(log_numerators)
        boltzman_probability = np.exp(boltzman_log_probs)

        bolts_act = np.random.choice(list(range(len(env.num_action_grid_map))), p=boltzman_probability)
        boltzman_rational_trajectory.append((current_state, bolts_act))
        probs = env.transitions[current_state][bolts_act]
        next_state = np.random.choice(env.num_states, p=probs)
        current_state = next_state

        # this if statement makes sure the temrinal state is included in the trajectory
        if current_state in env.terminals:
            boltzman_rational_trajectory.append((current_state, bolts_act))

    return boltzman_rational_trajectory



def generate_noisy_demo(env, eps=0, start_state=[0,0]):
    """
    Genarates noisy rational demonstrations consisting of state action pairs(s,a)
    :param env: Continuous Markov decision process passed by main see (markov_decision_process.py)
    :param beta: eps is a rationality quantification
    :param start_state: start state of demonstration
    :return:
    """
    current_state = start_state
    env.obs = np.array(current_state)
    max_traj = 200#env.num_states // 2  #this should be sufficiently long, maybe too long...
    noisy_rational_trajectory = []
    # q_values = calculate_q_values(env)
    term_state_set = [[env.goal[0]-0.01,env.goal[0]+0.01], [env.goal[1]-0.01,env.goal[1]+0.01]]

    while (
        # current_state not in env.terminals  #stop when we reach a terminal
        not (term_state_set[0][0]<current_state[0]<term_state_set[0][1]
            and term_state_set[1][0]<current_state[1]<term_state_set[1][1])
        and len(noisy_rational_trajectory) < max_traj
    ):  # need to add a trajectory length for infinite envs

       
        action = env.expert_policy(env.obs)
        # IPython.embed()
        obs, r, done, info = env.step(action, env.obs)
        current_state = env.obs
        
        noisy_rational_trajectory.append((current_state, action))
        env.obs = obs

        # this if statement makes sure the temrinal state is included in the trajectory
        if (term_state_set[0][0]<current_state[0]<term_state_set[0][1]
            and term_state_set[1][0]<current_state[1]<term_state_set[1][1]):
            noisy_rational_trajectory.append((current_state, action))

    return noisy_rational_trajectory



# Define the exponentiated quadratic 
def discretize_traj(env, traj):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)

    x_disc = np.array([round(i,2) for i in np.linspace(1,0, env.disc_x+1)])#np.arange(0, 1.0 + delta/2, delta)
    y_disc = np.array([round(i,2) for i in np.linspace(1,0, env.disc_y+1)])#np.arange(0, 1.0 + delta/2, delta)

    act_x_disc = np.array([round(i,2) for i in np.linspace(-np.pi,np.pi,env.disc_act_x+1)[:-1]])#np.arange(0, 2 * np.pi + delta/2, delta)
    act_y_disc = np.array([round(i,2) for i in np.linspace(env.max_speed,env.min_speed,env.disc_act_y+1)])#np.arange(0, max_speed + delta/2, delta)
   
    disc_traj = []
    for demo in traj:
        dist_x = (x_disc - demo[0][0])**2
        dist_y = (y_disc - demo[0][1])**2
        dist_act_x = (act_x_disc - demo[1][0])**2
        dist_act_y = (act_y_disc - demo[1][1])**2
        
        disc_traj.append(np.array([round(x_disc[np.argmin(dist_x)], 2), round(y_disc[np.argmin(dist_y)], 2),
        round(act_x_disc[np.argmin(dist_act_x)], 2), round(act_y_disc[np.argmin(dist_act_y)], 2)]))

    
    return disc_traj


# Define the exponentiated quadratic 
def FP_and_FN_and_TP(constraints, map_constr):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    TPR = np.sum(map_constr[np.nonzero(constraints)])/(len(np.nonzero(constraints)[0]))
    no_cnstr_ind = [i for i,k in enumerate(constraints) if k!=1]
    
    FPR = np.sum(map_constr[no_cnstr_ind])/len(no_cnstr_ind)
    
    FNR = np.sum(1-map_constr[np.nonzero(constraints)])/len(np.nonzero(constraints)[0])


    return TPR, FPR, FNR

def generate_weird_demo(env, start_state):
    current_state = start_state
    env.obs = np.array(current_state)
    max_traj = 200#env.num_states // 2  #this should be sufficiently long, maybe too long...
    noisy_rational_trajectory = []
    # q_values = calculate_q_values(env)
    term_state_set = [[env.goal[0]-0.01,env.goal[0]+0.01], [env.goal[1]-0.01,env.goal[1]+0.01]]
    # tt = [[0.01,0.1],[0.01,0.15],[0.02,0.2],[0.02,0.28],[0.02,0.34],[0.08,0.36],[0.15,0.4],[0.25,0.5]
    # ,[0.3,0.55],[0.2,0.65],[0.12,0.75],[0.03,0.85],[0.02,0.95],[0.01,0.99]]
    coin = np.random.rand()
    if coin <1/4:
        tt = [[0.9,0.05],[0.85,0.25],[0.8,0.3],[0.75,0.35],[0.7,0.4],[0.65,0.45],[0.6,0.5],[0.55,0.55]
        ,[0.5,0.6],[0.45,0.65],[0.4,0.7],[0.35,0.75],[0.3,0.8],[0.25,0.85],[0.2,0.9],[0.15,0.9],[0.07,0.95],[0.0,1]]
    elif coin>1/4 and coin<2/4:
        tt = [[0.9,0.01],[0.9,0.25],[0.8,0.3],[0.75,0.35],[0.75,0.4],[0.75,0.45],[0.7,0.5],[0.7,0.55]
        ,[0.7,0.6],[0.7,0.65],[0.65,0.7],[0.6,0.75],[0.55,0.8],[0.5,0.85],[0.45,0.9],[0.4,0.9],[0.3,0.95],[0.2,0.99],[0.1,0.99],[0.0,1]]
    elif coin>2/4 and coin<3/4:
        tt = [[1,0.05],[1,0.1],[1,0.2],[0.95,0.25],[0.85,0.3],[0.85,0.4],[0.8,0.5],[0.8,0.55]
        ,[0.7,0.6],[0.7,0.65],[0.65,0.7],[0.6,0.75],[0.55,0.8],[0.5,0.85],[0.45,0.9],[0.4,0.9],[0.3,0.95],[0.2,0.99],[0.1,0.99],[0.0,1]]
    else:
        tt = [[0.9,0.01],[0.85,0.05],[0.8,0.1],[0.75,0.1],[0.7,0.4],[0.65,0.15],[0.6,0.2],[0.55,0.25]
        ,[0.5,0.3],[0.45,0.35],[0.4,0.4],[0.35,0.45],[0.3,0.5],[0.25,0.55],[0.2,0.6],[0.15,0.65],[0.07,0.7],[0.01,0.75],[0.01,0.8],[0.01,0.9],[0.0,1]]
    for i in range(1,len(tt)):
        
        # print(env.obs)
        temp=[tt[i][0]-current_state[0],tt[i][1]-current_state[1]]
        act = env.cart_to_polar(temp)
        # act[0] =  act[0] + np.pi
        print(act[0])
        noisy_rational_trajectory.append((np.array(current_state),np.array([act[0],act[1]])))               
        current_state = current_state + env.polar_to_cart(act) + [np.random.rand()*0.05, np.random.rand()*0.05]        
       
    return noisy_rational_trajectory


