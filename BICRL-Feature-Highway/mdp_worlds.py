from mdp import HighwayMDP, MDP
import numpy as np
import IPython

def gen_simple_world():
    #four features, red (-1), blue (+5), white (0), yellow (+1)
    r = [1,0,0]
    b = [0,1,0]
    w = [0,0,1]
    gamma = 0.9
    #create state features for a 2x2 grid (really just an array, but I'm formating it to look like the grid world)
    state_features = [b, r, w, 
                      w, r, w,
                      w, w, w]
    feature_weights = [-1.0, 1.0, 0.0] #red feature has weight -1 and blue feature has weight +1
    
    noise = 0.0 #no noise in transitions
    env = FeatureMDP(3,3,[0],feature_weights, state_features, gamma, noise)
    return env


def random_gridworld(rows, columns):
    """
    Randomly chooses rewards, no terminal, noisy transition
    """
    random_mdp = MDP(rows, columns, [], np.random.randn(rows * columns), [], gamma=0.95, noise = 0.1)
    return random_mdp


def nonrand_gridworld(rows, columns, terminal_states, rewards , constraints, gamma, noise):
    """
    Randomly chooses rewards, no terminal, noisy transitions
    """
    # terminal_states = [columns - 1]#[rows * columns - 1]
    # rew = - np.ones(rows * columns)
    # rew[columns - 1] = 2.0
    # constraints = [4, 5, 6, 7]
    random_mdp = MDP(rows, columns, terminal_states, rewards, constraints, gamma, noise)
    return random_mdp

def highway(rows, columns, terminals, rewards, constraints, gamma, noise):
    """
    Randomly chooses rewards, no terminal, noisy transition
    """
    mdp = HighwayMDP(rows, columns, terminals, rewards, constraints=None, gamma=0.95, noise = 0.1)
    return mdp

