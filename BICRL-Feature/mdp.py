import IPython
import math
import copy
import numpy as np
import mdp_utils


class MDP:
    def __init__(self, num_rows, num_cols, terminals, rewards, constraints, gamma, noise=0.1):

        """
        Markov Decision Processes (MDP):
        num_rows: number of row in a environment
        num_cols: number of columns in environment
        terminals: terminal states (sink states)
        noise: with probability 2*noise the agent will move perpendicular to desired action split evenly, 
                e.g. if taking up action, then the agent has probability noise of going right and probability noise of going left.
        """
        self.gamma = gamma
        self.num_states = num_rows * num_cols 
        self.num_actions = 4  #up:0, down:1, left:2, right:3
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.terminals = terminals
        self.rewards = rewards  # think of this
        self.constraints = constraints
        self.state_grid_map = {}
        
        temp =  [(a, b) for b in np.arange(num_rows-1,-1,-1) for a in np.arange(num_cols)]
        for i in range(self.num_states):
            self.state_grid_map[i] = temp[i]
       
        
        #initialize transitions given desired noise level
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.init_transition_probabilities(noise)


    def init_transition_probabilities(self, noise):
        # 0: up, 1 : down, 2:left, 3:right

        UP = 0
        DOWN = 1
        LEFT = 2
        RIGHT = 3
        # going UP
        for s in range(self.num_states):

            # possibility of going foward

            if s >= self.num_cols:
                self.transitions[s][UP][s - self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][UP][s] = 1.0 - (2 * noise)

            # possibility of going left
            if s % self.num_cols == 0:
                self.transitions[s][UP][s] = noise
            else:
                self.transitions[s][UP][s - 1] = noise

            # possibility of going right

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][UP][s + 1] = noise
            else:
                self.transitions[s][UP][s] = noise

            # special case top left corner

            if s < self.num_cols and s % self.num_cols == 0.0:
                self.transitions[s][UP][s] = 1.0 - noise
            elif s < self.num_cols and s % self.num_cols == self.num_cols - 1:
                self.transitions[s][UP][s] = 1.0 - noise

        # going down
        for s in range(self.num_states):

            # self.num_rows = gridHeight
            # self.num_cols = gridwidth

            # possibility of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][DOWN][s + self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][DOWN][s] = 1.0 - (2 * noise)

            # possibility of going left
            if s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = noise
            else:
                self.transitions[s][DOWN][s - 1] = noise

            # possibility of going right
            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][DOWN][s + 1] = noise
            else:
                self.transitions[s][DOWN][s] = noise

            # checking bottom right corner
            if s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][DOWN][s] = 1.0 - noise
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][DOWN][s] = 1.0 - noise

        # going left
        # self.num_rows = gridHeight
        # self.num_cols = gridwidth
        for s in range(self.num_states):
            # possibility of going left

            if s % self.num_cols > 0:
                self.transitions[s][LEFT][s - 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][LEFT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][LEFT][s - self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # possiblity of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][LEFT][s + self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # check  top left corner
            if s < self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1.0 - noise
            elif s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1 - noise

        # going right
        for s in range(self.num_states):

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][RIGHT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][RIGHT][s - self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # possibility of going down

            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][RIGHT][s + self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # check top right corner
            if (s < self.num_cols) and (s % self.num_cols == self.num_cols - 1):
                self.transitions[s][RIGHT][s] = 1 - noise
            # check bottom rihgt corner case
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][RIGHT][s] = 1.0 - noise

        for s in range(self.num_states):
            if s in self.terminals:
                for a in range(self.num_actions):
                    for s2 in range(self.num_states):
                        self.transitions[s][a][s2] = 0.0

        # transition function not changed at all, just not allow action to take agent into a constraint
        # for s in range(self.num_states):        
        #     for a in range(self.num_actions):
        #         for s2 in range(self.num_states):
                    
        #             if s2 in np.nonzero(self.constraints)[0]:
        #                 self.transitions[s][a][s2] = 0.0

        # for s in range(self.num_states):
        #     if s in np.nonzero(self.constraints)[0]:
        #         for a in range(self.num_actions):
        #             for s2 in range(self.num_states):
        #                 self.transitions[s][a][s2] = 0.0

    
    def set_rewards(self, _rewards):
        self.rewards = _rewards

    def set_constraints(self, _constraints):
        self.constraints = _constraints

    def set_gamma(self, gamma):
        assert(gamma < 1.0 and gamma > 0.0)
        self.gamma = gamma


class HighwayMDP:
    def __init__(self, num_rows, num_cols, terminals, rewards, constraints, gamma, noise=0.1):

        """
        Markov Decision Processes for Highway Environment:
        num_lanes: lanes on the highway
        length_highway: length of street
        terminals: terminal states (sink states)
        noise: with probability 2*noise the agent will move perpendicular to desired action split evenly, 
                e.g. if taking up action, then the agent has probability noise of going right and probability noise of going left.
        """
        self.gamma = gamma
        self.num_states = num_rows * num_cols 
        self.num_actions = 3  #up:0, left:1, right:2
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.terminals = terminals
        self.rewards = rewards
        self.state_grid_map = {}
        
        temp =  [(a, b) for b in np.arange(num_rows-1,-1,-1) for a in np.arange(num_cols)]
        for i in range(self.num_states):
            self.state_grid_map[i] = temp[i]
        # 0- no tailgaiting
        # 1- at least one grid cell to the left of the biker
        # 2- overtake only form the left
        self.feature_names = ['tail_gate', 'close_biker',  'car_on_left', 'car_on_right',
        'left_lane','middle_lane','right_lane', 
        'occupied_cell','goal']#["stone", "grass", "car", "ped", "term"]#, "HOV", "car-in-f", "ped-in-f", "police"]
        self.num_features = len(self.feature_names)
        self.real_constraints = [0,1,2,self.num_features-1]
        
        # self.W = np.array([-1, -0.5, -5, -10, 5])#, -2, -5, 0])
        self.get_feature_matrix()

        #initialize transitions given desired noise level
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.init_transition_probabilities(noise)


    def get_feature_matrix(self):
        self.feature_matrix = np.zeros((self.num_states, self.num_features))
        # for i in range(self.num_cols):
        #     self.feature_matrix[i][-1] = 1
        # no tailgaiting any car, pass at least one grid cell away from the rider, pass only from the left
        # features:
        # 0- 'tail_gate'
        # 1- 'close_biker'
        # 2- 'car_on_left'
        # 3- 'car_on_right'
        # 4- 'left_lane'
        # 5- 'middle_lane'
        # 6- 'right_lane'
        # 7- 'occupied cell'
        # 8- 'goal'
       
        self.feature_matrix[0*self.num_cols+0][3] = 1# car on right
        self.feature_matrix[0*self.num_cols+2][2] = 1# car on left
        self.feature_matrix[1*self.num_cols+1][0] = 1# no tailgating
        
        self.feature_matrix[2*self.num_cols+1][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[2*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[3*self.num_cols+1][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[4*self.num_cols+1][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[4*self.num_cols+2][1] = 1# leave at least one grid distance from the rider

        self.feature_matrix[4*self.num_cols+1][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[4*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[5*self.num_cols+1][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[6*self.num_cols+1][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[6*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
       

        self.feature_matrix[9*self.num_cols+0][0] = 1# no tailgate  # comm out 
        self.feature_matrix[8*self.num_cols+1][2] = 1# car on left  # comm out 

        # self.feature_matrix[9*self.num_cols+2][0] = 1# no tailgate  # comm out  new
        # self.feature_matrix[8*self.num_cols+1][2] = 1# car on left  # comm out 
        # self.feature_matrix[8*self.num_cols+2][2] = 1# car on left

        self.feature_matrix[0*self.num_cols+1][7] = 1# occupied cell
        self.feature_matrix[3*self.num_cols+2][7] = 1# occupied cell
        self.feature_matrix[8*self.num_cols+0][7] = 1# occupied cell  # comm out 

        self.feature_matrix[5*self.num_cols+2][7] = 1# occupied cell

        # self.feature_matrix[8*self.num_cols+2][7] = 1# occupied cell  # comm out new
       
        self.feature_matrix[0][8] = 1# goal cell

        for i in range(self.num_rows):
            self.feature_matrix[i*self.num_cols+0][4] = 1
            self.feature_matrix[i*self.num_cols+1][5] = 1
            self.feature_matrix[i*self.num_cols+2][6] = 1
        
        # self.feature_matrix[:, self.num_features] = 1
        
    def get_rewards(self, W, rewards):
        # obtain reward from feature vectors
        rewards_temp = np.matmul(self.feature_matrix, W)
        return rewards_temp


    def init_transition_probabilities(self, noise):
        # 0: up, 1 : down, 2:left, 3:right

        UP = 0
        LEFT = 1
        RIGHT = 2
        # going UP
        for s in range(self.num_states):

            # possibility of going foward

            if s >= self.num_cols:
                self.transitions[s][UP][s - self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][UP][s] = 1.0 - (2 * noise)

            # possibility of going left
            if s % self.num_cols == 0:
                self.transitions[s][UP][s] = noise
            else:
                self.transitions[s][UP][s - 1] = noise

            # possibility of going right

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][UP][s + 1] = noise
            else:
                self.transitions[s][UP][s] = noise

            # special case top left corner

            if s < self.num_cols and s % self.num_cols == 0.0:
                self.transitions[s][UP][s] = 1.0 - noise
            elif s < self.num_cols and s % self.num_cols == self.num_cols - 1:
                self.transitions[s][UP][s] = 1.0 - noise

        # going left
        # self.num_rows = gridHeight
        # self.num_cols = gridwidth
        for s in range(self.num_states):
            # possibility of going left

            if s % self.num_cols > 0:
                self.transitions[s][LEFT][s - 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][LEFT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][LEFT][s - self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # possiblity of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][LEFT][s + self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # check  top left corner
            if s < self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1.0 - noise
            elif s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1 - noise

        # going right
        for s in range(self.num_states):

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][RIGHT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][RIGHT][s - self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # possibility of going down

            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][RIGHT][s + self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # check top right corner
            if (s < self.num_cols) and (s % self.num_cols == self.num_cols - 1):
                self.transitions[s][RIGHT][s] = 1 - noise
            # check bottom rihgt corner case
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][RIGHT][s] = 1.0 - noise

        for s in range(self.num_states):
            if s in self.terminals:
                for a in range(self.num_actions):
                    for s2 in range(self.num_states):
                        self.transitions[s][a][s2] = 0.0
    
    def set_rewards(self, _rewards):
        self.rewards = _rewards


   
class HighwayMDP_test:
    def __init__(self, num_rows, num_cols, terminals, rewards, constraints, gamma, noise=0.1):

        """
        Markov Decision Processes for Highway Environment:
        num_lanes: lanes on the highway
        length_highway: length of street
        terminals: terminal states (sink states)
        noise: with probability 2*noise the agent will move perpendicular to desired action split evenly, 
                e.g. if taking up action, then the agent has probability noise of going right and probability noise of going left.
        """
        self.gamma = gamma
        self.num_states = num_rows * num_cols 
        self.num_actions = 3  #up:0, left:1, right:2
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.terminals = terminals
        self.rewards = rewards
        self.state_grid_map = {}
        
        temp =  [(a, b) for b in np.arange(num_rows-1,-1,-1) for a in np.arange(num_cols)]
        for i in range(self.num_states):
            self.state_grid_map[i] = temp[i]
        # 0- no tailgaiting
        # 1- at least one grid cell to the left of the biker
        # 2- overtake only form the left
        self.feature_names = ['tail_gate', 'close_biker',  'car_on_left', 'car_on_right',
        'left_lane','middle_lane','right_lane', 
        'occupied_cell','goal']#["stone", "grass", "car", "ped", "term"]#, "HOV", "car-in-f", "ped-in-f", "police"]
        self.num_features = len(self.feature_names)
        self.real_constraints = [0,1,2,self.num_features-1]
        
        # self.W = np.array([-1, -0.5, -5, -10, 5])#, -2, -5, 0])
        self.get_feature_matrix()

        #initialize transitions given desired noise level
        self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.init_transition_probabilities(noise)


    def get_feature_matrix(self):
        self.feature_matrix = np.zeros((self.num_states, self.num_features))
        # for i in range(self.num_cols):
        #     self.feature_matrix[i][-1] = 1
        # no tailgaiting any car, pass at least one grid cell away from the rider, pass only from the left
        # features:
        # 0- 'tail_gate'
        # 1- 'close_biker'
        # 2- 'car_on_left'
        # 3- 'car_on_right'
        # 4- 'left_lane'
        # 5- 'middle_lane'
        # 6- 'right_lane'
        # 7- 'occupied cell'
        # 8- 'goal'
        # 1st car
        self.feature_matrix[0*self.num_cols+0][3] = 1# car on right
        self.feature_matrix[0*self.num_cols+2][2] = 1# car on left
        self.feature_matrix[1*self.num_cols+1][0] = 1# no tailgating
        # 2nd car
        self.feature_matrix[2*self.num_cols+0][3] = 1# car on right
        self.feature_matrix[2*self.num_cols+2][2] = 1# car on left
        self.feature_matrix[3*self.num_cols+1][0] = 1# no tailgating
        # 3rd car
        self.feature_matrix[9*self.num_cols+2][3] = 1# car on right
        self.feature_matrix[10*self.num_cols+3][0] = 1# no tailgating
        # 4th car
        self.feature_matrix[2*self.num_cols+2][3] = 1# car on right
        self.feature_matrix[3*self.num_cols+3][0] = 1# no tailgating

        # 1st rider
        self.feature_matrix[5*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[5*self.num_cols+3][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[6*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[7*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[7*self.num_cols+3][1] = 1# leave at least one grid distance from the rider

        # 2nd rider
        self.feature_matrix[8*self.num_cols+0][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[8*self.num_cols+1][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[9*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[10*self.num_cols+0][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[10*self.num_cols+1][1] = 1# leave at least one grid distance from the rider

        # 3nd rider
        self.feature_matrix[11*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[11*self.num_cols+3][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[12*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[13*self.num_cols+2][1] = 1# leave at least one grid distance from the rider
        self.feature_matrix[13*self.num_cols+3][1] = 1# leave at least one grid distance from the rider
       

        # self.feature_matrix[9*self.num_cols+0][0] = 1# no tailgate
        # self.feature_matrix[8*self.num_cols+1][2] = 1# car on left
        # self.feature_matrix[8*self.num_cols+2][2] = 1# car on left

        self.feature_matrix[0*self.num_cols+1][7] = 1# occupied cell
        self.feature_matrix[2*self.num_cols+1][7] = 1# occupied cell
        self.feature_matrix[6*self.num_cols+3][7] = 1# occupied cell
        self.feature_matrix[9*self.num_cols+0][7] = 1# occupied cell
        self.feature_matrix[2*self.num_cols+3][7] = 1# occupied cell
        self.feature_matrix[12*self.num_cols+3][7] = 1# occupied cell
        self.feature_matrix[9*self.num_cols+3][7] = 1# occupied cell
       
        self.feature_matrix[0][8] = 1# goal cell

        # for i in range(self.num_rows):
        #     self.feature_matrix[i*self.num_cols+0][4] = 1
        #     self.feature_matrix[i*self.num_cols+1][5] = 1
        #     self.feature_matrix[i*self.num_cols+2][6] = 1
            
        
        # self.feature_matrix[:, self.num_features] = 1
        
    def get_rewards(self, W, rewards):
        # obtain rewards from feature vectors
        rewards_temp = np.matmul(self.feature_matrix, W)
        return rewards_temp


    def init_transition_probabilities(self, noise):
        # 0: up, 1 : down, 2:left, 3:right

        UP = 0
        LEFT = 1
        RIGHT = 2
        # going UP
        for s in range(self.num_states):

            # possibility of going foward

            if s >= self.num_cols:
                self.transitions[s][UP][s - self.num_cols] = 1.0 - (2 * noise)
            else:
                self.transitions[s][UP][s] = 1.0 - (2 * noise)

            # possibility of going left
            if s % self.num_cols == 0:
                self.transitions[s][UP][s] = noise
            else:
                self.transitions[s][UP][s - 1] = noise

            # possibility of going right

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][UP][s + 1] = noise
            else:
                self.transitions[s][UP][s] = noise

            # special case top left corner

            if s < self.num_cols and s % self.num_cols == 0.0:
                self.transitions[s][UP][s] = 1.0 - noise
            elif s < self.num_cols and s % self.num_cols == self.num_cols - 1:
                self.transitions[s][UP][s] = 1.0 - noise

        # going left
        # self.num_rows = gridHeight
        # self.num_cols = gridwidth
        for s in range(self.num_states):
            # possibility of going left

            if s % self.num_cols > 0:
                self.transitions[s][LEFT][s - 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][LEFT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][LEFT][s - self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # possiblity of going down
            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][LEFT][s + self.num_cols] = noise
            else:
                self.transitions[s][LEFT][s] = noise

            # check  top left corner
            if s < self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1.0 - noise
            elif s >= (self.num_rows - 1) * self.num_cols and s % self.num_cols == 0:
                self.transitions[s][LEFT][s] = 1 - noise

        # going right
        for s in range(self.num_states):

            if s % self.num_cols < self.num_cols - 1:
                self.transitions[s][RIGHT][s + 1] = 1.0 - (2 * noise)
            else:
                self.transitions[s][RIGHT][s] = 1.0 - (2 * noise)

            # possibility of going up

            if s >= self.num_cols:
                self.transitions[s][RIGHT][s - self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # possibility of going down

            if s < (self.num_rows - 1) * self.num_cols:
                self.transitions[s][RIGHT][s + self.num_cols] = noise
            else:
                self.transitions[s][RIGHT][s] = noise

            # check top right corner
            if (s < self.num_cols) and (s % self.num_cols == self.num_cols - 1):
                self.transitions[s][RIGHT][s] = 1 - noise
            # check bottom rihgt corner case
            elif (
                s >= (self.num_rows - 1) * self.num_cols
                and s % self.num_cols == self.num_cols - 1
            ):
                self.transitions[s][RIGHT][s] = 1.0 - noise

        for s in range(self.num_states):
            if s in self.terminals:
                for a in range(self.num_actions):
                    for s2 in range(self.num_states):
                        self.transitions[s][a][s2] = 0.0
    
    def set_rewards(self, _rewards):
        self.rewards = _rewards


   