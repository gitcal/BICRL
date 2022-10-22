import IPython
import math
import copy
import numpy as np
import mdp_utils
from gym import spaces
from scipy.stats import bernoulli


# this is cont
class PointMass2D_v2():
	"""Navigate to target while avoiding trap
	Not allowed to go out of bounds (unit square)
	Episode terminates upon reaching target or trap
	Continuous actions (velocity vector)
	2D position observations
	"""
	def __init__(self,
			   disc_x=10,
			   disc_y=10,
			   disc_act_x=16,
			   disc_act_y=1,
			   noise= 0.0,
			   max_ep_len=200,
			   goal_dist_thresh=0.1,
			   goal_dist_thresh_shaping=0.25,
			   trap_dist_thresh=0.2,
			   succ_rew_bonus=1.,
			   crash_rew_penalty=-10.,
			   max_speed=0.25,
			   goal=None,
			   trap=None,
			   init_pos=None):	

		if goal is None:
		  goal = np.array([0.5, 1.0])

		if trap is None:
		  trap1 = np.array([0.5, 0.5])
		 
		self.max_ep_len = max_ep_len
		self.noise = noise
		self.goal_dist_thresh = goal_dist_thresh
		self.trap_dist_thresh = trap_dist_thresh
		self.succ_rew_bonus = succ_rew_bonus
		self.goal_dist_thresh_shaping = goal_dist_thresh_shaping
		self.crash_rew_penalty = crash_rew_penalty
		self.max_speed = max_speed
		self.min_speed = 0.02
		self.init_pos = init_pos
		self.goal = goal
		self.trap = [trap1]#, trap2]
		self.timestep = 0
		self.disc_x = disc_x #20
		self.disc_y = disc_y #20
		self.disc_act_x = disc_act_x #8
		self.disc_act_y = disc_act_y #3
		self.gamma = 0.95
		self.num_states = (self.disc_x + 1) * (self.disc_y + 1)
		self.num_actions = (self.disc_act_x) * (self.disc_act_y + 1)
		self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
		self.state_grid_map = {}
		self.num_state_grid_map = {}
		self.action_grid_map = {}
		self.num_action_grid_map = {}
		self.rewards = np.zeros(self.num_states)
		self.target_rew = 5
		self.l_rew = -2

		self.terminals = [60,61,72,73]#[55,66,44]#10x10[65, 78, 91]#65, 78, 91 12x12 ###[137, 135, 120, 121, 153, 154]#16x16

		self.n_act_dim = 2  # angle, speed
		self.n_obs_dim = 2  # position
		self.observation_space = spaces.Box(
			np.zeros(self.n_obs_dim), np.ones(self.n_obs_dim))
		self.action_space = spaces.Box(
			np.zeros(2), np.array([2 * np.pi, self.max_speed]))
		self.name = 'pointmass'
		# self.expert_policy = self.make_expert_policy()
		# self.subopt_policy = self.make_noisy_expert_policy(eps=0.75) ###################### uncom
		self.delta = 0.1
		self.default_init_obs = init_pos if init_pos is not None else np.zeros(2)
		self.rew_classes = np.array(
			[self.crash_rew_penalty, 0., self.succ_rew_bonus])
		self.only_terminal_reward = True
		self.pos = None
		
		temp =  [[round(a, 2), round(b, 2)] for a in np.linspace(1,0,self.disc_x+1) for b in np.linspace(1,0,self.disc_y+1)]
		for i in range(self.num_states):
			self.num_state_grid_map[i] = temp[i]
			self.state_grid_map[tuple(temp[i])] = i
		
		temp =  [[round(a, 2), round(b, 2)] for a in np.linspace(-np.pi,np.pi,self.disc_act_x+1)[:-1] for b in np.linspace(self.max_speed,self.min_speed,self.disc_act_y+1)]
		# The [:-1] so that 2pi and 0 are not both inlcuded
		
		for i in range(self.num_actions):
			self.num_action_grid_map[i] = temp[i]
			self.action_grid_map[tuple(temp[i])] = i
		self.state_size = 2 # this is for input of neural network node size
		self.action_size = len(self.action_grid_map) # this is for out of 
		
		self.init_transition_probabilities()



	def init_transition_probabilities(self, noise=0):
		# 0: up, 1 : down, 2:left, 3:right

		# UP = 0
		# DOWN = 1
		# LEFT = 2
		# RIGHT = 3
		# going UP
		
		for s in range(self.num_states):

			for a in range(self.num_actions):
				action = np.array(self.num_action_grid_map[a])
				# action = self.polar_to_cart(self.normalize_polar(action))
				start_pos = np.array(self.num_state_grid_map[s])
				# if self.state_grid_map[s] == [0.88, 0.38]:
				
				self.pos = start_pos
				end_pos_tup = self.step(action)

				if end_pos_tup == None:

					end_pos_tup = start_pos
					demo = [(end_pos_tup, action)]
				else:
					demo = [(end_pos_tup[0], action)]


				disc_state_action = mdp_utils.discretize_traj(self, demo)
				key = self.state_grid_map[tuple(disc_state_action[0][0:2])]
		
				temp_list = list(self.num_state_grid_map.values())
				dxdy = np.round(np.abs(temp_list[-1][1]-temp_list[-2][1]),2)
				new_state = self.num_state_grid_map[key]
				step = self.disc_x+1

				if action[0]>np.pi/4 and action[0]<3*np.pi/4:#up
					if new_state[0]==1.0: 
						self.transitions[s][a][key] = 1-self.noise
						dxdy = self.num_state_grid_map[key+step][0]
						temp_state1 = copy.deepcopy(new_state)
						temp_state1[0] = dxdy#np.round(temp_state1[0]-dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						self.transitions[s][a][key1] = self.noise

					elif new_state[0]==0.0:
						self.transitions[s][a][key] = 1-self.noise
						dxdy = self.num_state_grid_map[key-step][0]
						temp_state1 = copy.deepcopy(new_state)
						temp_state1[0] = dxdy#np.round(temp_state1[0]+dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						self.transitions[s][a][key1] = self.noise						

					else:
						self.transitions[s][a][key] = 1-2*self.noise
						temp_state1 = copy.deepcopy(new_state)
						dxdy = self.num_state_grid_map[key+step][0]
						temp_state1[0] = dxdy#np.round(temp_state1[0]-dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						temp_state2 = copy.deepcopy(new_state)
						dxdy = self.num_state_grid_map[key-step][0]
						temp_state2[0] = dxdy#np.round(temp_state2[0]+dxdy,2)
						key2 = self.state_grid_map[tuple(temp_state2)]
						self.transitions[s][a][key1] = self.noise
						self.transitions[s][a][key2] = self.noise

				elif action[0]>-3*np.pi/4 and action[0]<-np.pi/4:# down
					if new_state[0]==1.0: 
						self.transitions[s][a][key] = 1-self.noise
						dxdy = self.num_state_grid_map[key+step][0]
						temp_state1 = copy.deepcopy(new_state)
						temp_state1[0] = dxdy#np.round(temp_state1[0]-dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						self.transitions[s][a][key1] = self.noise

					elif new_state[0]==0.0:
						self.transitions[s][a][key] = 1-self.noise
						dxdy = self.num_state_grid_map[key-step][0]
						temp_state1 = copy.deepcopy(new_state)
						temp_state1[0] = dxdy#np.round(temp_state1[0]+dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						self.transitions[s][a][key1] = self.noise						

					else:
						self.transitions[s][a][key] = 1-2*self.noise
						temp_state1 = copy.deepcopy(new_state)
						dxdy = self.num_state_grid_map[key+step][0]
						temp_state1[0] = dxdy#np.round(temp_state1[0]-dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						temp_state2 = copy.deepcopy(new_state)
						dxdy = self.num_state_grid_map[key-step][0]
						temp_state2[0] = dxdy#np.round(temp_state2[0]+dxdy,2)
						key2 = self.state_grid_map[tuple(temp_state2)]
						self.transitions[s][a][key1] = self.noise
						self.transitions[s][a][key2] = self.noise

				elif action[0]>-np.pi/4 and action[0]<np.pi/4:# right
					if new_state[1]==1.0: 
						self.transitions[s][a][key] = 1-self.noise
						dxdy = self.num_state_grid_map[key+1][1]
						temp_state1 = copy.deepcopy(new_state)
						temp_state1[1] = dxdy#np.round(temp_state1[1]-dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						self.transitions[s][a][key1] = self.noise

					elif new_state[1]==0.0:
						self.transitions[s][a][key] = 1-self.noise
						dxdy = self.num_state_grid_map[key-1][1]
						temp_state1 = copy.deepcopy(new_state)
						temp_state1[1] = dxdy#np.round(temp_state1[1]+dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						self.transitions[s][a][key1] = self.noise						

					else:
						self.transitions[s][a][key] = 1-2*self.noise
						temp_state1 = copy.deepcopy(new_state)
						dxdy = self.num_state_grid_map[key+1][1]
						temp_state1[1] = dxdy#np.round(temp_state1[1]-dxdy,2)						
						# IPython.embed()
						key1 = self.state_grid_map[tuple(temp_state1)]
						temp_state2 = copy.deepcopy(new_state)
						dxdy = self.num_state_grid_map[key-1][1]
						temp_state2[1] = dxdy#np.round(temp_state2[1]+dxdy,2)
						key2 = self.state_grid_map[tuple(temp_state2)]
						self.transitions[s][a][key1] = self.noise
						self.transitions[s][a][key2] = self.noise

				elif action[0]>3*np.pi/4 or action[0]<-3*np.pi/4:# left

					if new_state[1]==1.0:
						# IPython.embed()	 
						self.transitions[s][a][key] = 1-self.noise
						# print('here')
						# IPython.embed()
						dxdy = self.num_state_grid_map[key+1][1]
						temp_state1 = copy.deepcopy(new_state)
						temp_state1[1] = dxdy#np.round(temp_state1[1]-dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						self.transitions[s][a][key1] = self.noise

					elif new_state[1]==0.0:
						self.transitions[s][a][key] = 1-self.noise
						dxdy = self.num_state_grid_map[key-1][1]
						temp_state1 = copy.deepcopy(new_state)
						temp_state1[1] = dxdy#np.round(temp_state1[1]+dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						self.transitions[s][a][key1] = self.noise						

					else:
						self.transitions[s][a][key] = 1-2*self.noise
						temp_state1 = copy.deepcopy(new_state)
						dxdy = self.num_state_grid_map[key+1][1]
						temp_state1[1] = dxdy#np.round(temp_state1[1]-dxdy,2)
						key1 = self.state_grid_map[tuple(temp_state1)]
						temp_state2 = copy.deepcopy(new_state)
						dxdy = self.num_state_grid_map[key-1][1]
						temp_state2[1] = dxdy#np.round(temp_state2[1]+dxdy,2)
						key2 = self.state_grid_map[tuple(temp_state2)]
						self.transitions[s][a][key1] = self.noise
						self.transitions[s][a][key2] = self.noise

				# self.transitions[s][a][key] = 1
				self.rewards[s] = self.l_rew
#r
		
		for i in self.terminals:
			# if rew == 2:
			self.rewards[i] = self.target_rew
			# cnt +=1 
	
		# self.terminals = terminals
		for s in range(self.num_states):
			if s in self.terminals:
				for a in range(self.num_actions):
					for s2 in range(self.num_states):
						self.transitions[s][a][s2] = 0
						# if a==26:
						# 	self.transitions[s][a][s2] = 1

	def get_actions_from_human_demos(self, demos):

		act_list = []
		for traj in demos:
			act_list_temp = []
			for k in range(len(traj)-1):
				diff = traj[k+1]-traj[k]
				act_temp = self.cart_to_polar(diff)				
				act_list_temp.append([traj[k],act_temp])
			act_list_temp.append([traj[k+1], np.array([np.pi/2, 0.1])])# add last action
			act_list.append(act_list_temp)
		return act_list


				
	def find_teminals(self):
		terminals = []
		cnt = 0
		for rew in self.rewards:
			if rew == 2:
				terminals.append(cnt)
			cnt += 1
		return terminals

   
	def set_rewards(self, _rewards):
		self.rewards = _rewards

	def set_constraints(self, _constraints):
		self.constraints = _constraints

	def set_gamma(self, gamma):
		assert(gamma < 1.0 and gamma > 0.0)
		self.gamma = gamma

	def prob_succ(self, obses):
		at_goal = np.linalg.norm(obses - self.goal, axis=1) <= self.goal_dist_thresh
		return at_goal.astype(float)

	def prob_succ_shaping_rew(self, obses):
		at_goal = np.linalg.norm(obses - self.goal, axis=1) <= self.goal_dist_thresh_shaping
		return at_goal.astype(float)

	def prob_crash(self, obses):
		at_trap = (np.linalg.norm(obses - self.trap[0], axis=1) <= self.trap_dist_thresh)# or (np.linalg.norm(obses - self.trap[1], axis=1) <= self.trap_dist_thresh)
		return at_trap.astype(float)

	def reward_func(self, obses, acts, next_obses):
	
		# if self.prob_succ_shaping_rew(next_obses):		
		r = np.array([0.0])
		if self.prob_succ(next_obses):
			r += 20 * self.succ_rew_bonus #* self.prob_succ(next_obses)
		elif self.prob_crash(next_obses):
			r += self.crash_rew_penalty
		else:
			r += 0
		
		return r

	def obs(self):
		return self.pos

	def cart_to_polar(self, v):
		return np.array([np.arctan2(v[1], v[0]), np.linalg.norm(v)])

	def normalize_ang(self, a):
		return (2 * np.pi - abs(a) % (2 * np.pi)) if a < 0 else (abs(a) %
															 (2 * np.pi))

	def normalize_polar(self, v):
		# print(v)
		return np.array([self.normalize_ang(v[0]), min(v[1], self.max_speed)])

	def polar_to_cart(self, v):
		return v[1] * np.array([np.cos(v[0]), np.sin(v[0])])

	def step(self, action):
		# 0 degrees goes right
		# pi/2 goes up
		# pi goes left
		# 3pi/2 gown down
		old_pos = self.pos
		action = self.polar_to_cart(self.normalize_polar(action))
		
		if (old_pos + action >= 0).all() and (old_pos + action <=
											   1).all():  # stay in bounds
		   pos = old_pos + action
		   self.pos = pos
		else:
		   self.pos = old_pos
		   r = self.reward_func(old_pos[np.newaxis, :], action[np.newaxis, :],
							 old_pos[np.newaxis, :])[0]
		   return old_pos, r, False, {}

		self.succ = np.linalg.norm(pos - self.goal) <= self.goal_dist_thresh
		self.crash = (np.linalg.norm(pos - self.trap[0]) <= self.trap_dist_thresh) #or (np.linalg.norm(pos - self.trap[1]) <= self.trap_dist_thresh)

		self.timestep += 1

		# obs = self.obs()
		# r = self.reward_func(self.prev_obs[np.newaxis, :], action[np.newaxis, :],
		#                      obs[np.newaxis, :])[0]
		r = self.reward_func(old_pos[np.newaxis, :], action[np.newaxis, :],
							 pos[np.newaxis, :])[0]
		done = self.succ# or self.crash
		info = {'goal': self.goal, 'succ': self.succ, 'crash': self.crash}

		# self.prev_obs = obs

		return pos, r, done, info

	def reset(self):
		self.pos = np.array([0.5,0.05])
		self.pos=np.array([  np.random.rand(), 0.05 + 0.2 * np.random.rand()])#np.random.rand()])#
		# coin = np.random.rand()
		# if coin<1/3:
		# 	self.pos=np.array([ np.random.rand(), 0.05 + 0.2 * np.random.rand()])
		# elif coin>1/3 and coin<2/3:
		# 	self.pos=np.array([ 0.01 + 0.2 * np.random.rand(), 0.5 + 0.3 * np.sign(np.random.randn())* np.random.rand()])
		# else:
		# 	self.pos=np.array([ 1- 0.01 - 0.2 * np.random.rand(), 0.5 + 0.3 * np.sign(np.random.randn())* np.random.rand()])
		# self.pos=np.array([np.random.rand(), np.random.rand()])
		return self.pos



  

