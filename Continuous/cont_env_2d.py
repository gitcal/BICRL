import IPython
import math
import copy
import numpy as np
import mdp_utils
from gym import spaces


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


   
   
	def set_rewards(self, _rewards):
		self.rewards = _rewards

	def set_constraints(self, _constraints):
		self.constraints = _constraints

	def set_gamma(self, gamma):
		assert(gamma < 1.0 and gamma > 0.0)
		self.gamma = gamma


class PointMass2D():
	"""Navigate to target while avoiding trap
	Not allowed to go out of bounds (unit square)
	Episode terminates upon reaching target or trap
	Continuous actions (velocity vector)
	2D position observations
	"""
	def __init__(self,
			   max_ep_len=1000,
			   goal_dist_thresh=0.25,
			   trap_dist_thresh=0.25,
			   succ_rew_bonus=1.,
			   crash_rew_penalty=-10.,
			   max_speed=0.2,
			   goal=None,
			   trap=None,
			   init_pos=None):	

		if goal is None:
		  goal = np.array([0.0, 1.0])

		if trap is None:
		  trap1 = np.array([0.0, 0.45])
		  trap2 = np.array([1, 0.45])
		self.max_ep_len = max_ep_len
		self.goal_dist_thresh = goal_dist_thresh
		self.trap_dist_thresh = trap_dist_thresh
		self.succ_rew_bonus = succ_rew_bonus
		self.crash_rew_penalty = crash_rew_penalty
		self.max_speed = max_speed
		self.init_pos = init_pos
		self.goal = goal
		self.trap = [trap1,trap2]
		self.timestep = 0
		self.disc_x = 6
		self.disc_y = 6
		self.disc_act_x = 8
		self.disc_act_y = 3
		self.gamma = 0.95
		self.num_states = (self.disc_x + 1) * (self.disc_y + 1)
		self.num_actions = (self.disc_act_x) * (self.disc_act_y + 1)
		self.transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
		self.state_grid_map = {}
		self.action_grid_map = {}
		self.rewards = np.zeros(self.num_states)
		# non-overlapping target/trap
		if (np.linalg.norm(self.goal - self.trap[0]) < 2 * self.goal_dist_thresh) or (np.linalg.norm(self.goal - self.trap[1]) < 2 * self.goal_dist_thresh):
		  raise ValueError

		self.n_act_dim = 2  # angle, speed
		self.n_obs_dim = 2  # position
		self.observation_space = spaces.Box(
			np.zeros(self.n_obs_dim), np.ones(self.n_obs_dim))
		self.action_space = spaces.Box(
			np.zeros(2), np.array([2 * np.pi, self.max_speed]))
		self.name = 'pointmass'
		self.expert_policy = self.make_expert_policy()
		# self.subopt_policy = self.make_noisy_expert_policy(eps=0.75) ###################### uncom
		self.delta = 0.1
		self.default_init_obs = init_pos if init_pos is not None else np.zeros(2)
		self.rew_classes = np.array(
			[self.crash_rew_penalty, 0., self.succ_rew_bonus])
		self.only_terminal_reward = True
		self.pos = None

		temp =  [[round(a, 2), round(b, 2)] for a in np.linspace(1,0,self.disc_x+1) for b in np.linspace(1,0,self.disc_y+1)]
		for i in range(self.num_states):
			self.state_grid_map[i] = temp[i]
		# IPython.embed()
		temp =  [[round(a, 2), round(b, 2)] for a in np.linspace(-np.pi,np.pi,self.disc_act_x+1)[:-1] for b in np.linspace(self.max_speed,0,self.disc_act_y+1)]
		# The [:-1] so that 2pi and 0 are not both inlcuded
		for i in range(self.num_actions):
			self.action_grid_map[i] = temp[i]
		# self.terminals = self.find_teminals()
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
				action = np.array(self.action_grid_map[a])
				# action = self.polar_to_cart(self.normalize_polar(action))
				start_pos = np.array(self.state_grid_map[s])
				# if self.state_grid_map[s] == [0.88, 0.38]:
				# 	IPython.embed()
				end_pos_tup = self.step(action, start_pos)


				if end_pos_tup == None:
					end_pos_tup = start_pos
					demo = [(end_pos_tup, action)]
					r = 0
				else:
					demo = [(end_pos_tup[0], action)]
					# if np.abs(start_pos[1]-end_pos_tup[0][1])>0.05:
					# 	# IPython.embed()
					# 	print(start_pos,end_pos_tup[0])
					r = end_pos_tup[1]
					# print(start_pos,end_pos_tup[0])
				# print(start_pos,end_pos_tup[0])
				# demo = [(end_pos_tup[0], action)]

				disc_state_action = mdp_utils.discretize_traj(self, demo, self.delta, self.max_speed)
				# print('here')
				key = None
				for key_temp, val in self.state_grid_map.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
					# if np.array_equal(val, disc_state_action[0][0:2]):
					# 	key = key_temp
					if val == [round(disc_state_action[0][0:2][0],2),round(disc_state_action[0][0:2][1],2)]:
						key=key_temp
				if key == None:
					print('here')
					IPython.embed()
				self.transitions[s][a][key] = 1
				self.rewards[s] = r 
		terminals = []
		cnt = 0
		for rew in self.rewards:
			if rew == 2:
				terminals.append(cnt)
		IPython.embed()
		# cnt = 0
		# for rew in self.rewards:
		# 	if rew == 2:
		# 		self.rewards[cnt] = 0
		# 	cnt += 1
		# self.rewards[72] = 2
		# terminals.append(72)
		self.terminals = terminals
		for s in range(self.num_states):
			if s in terminals:
				for a in range(self.num_actions):
					for s2 in range(self.num_states):
						self.transitions[s][a][s2] = 0

				# print(s,a,key)
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

	def prob_crash(self, obses):
		at_trap = (np.linalg.norm(obses - self.trap[0], axis=1) <= self.trap_dist_thresh) or (np.linalg.norm(obses - self.trap[1], axis=1) <= self.trap_dist_thresh)
		return at_trap.astype(float)

	def reward_func(self, obses, acts, next_obses):
		r = 2 * self.succ_rew_bonus * self.prob_succ(next_obses)
		r += self.crash_rew_penalty * self.prob_crash(next_obses)
		return r

	def obs(self):
		return self.pos

	def cart_to_polar(self, v):
		return np.array([np.arctan2(v[1], v[0]), np.linalg.norm(v)])

	def normalize_ang(self, a):
		return (2 * np.pi - abs(a) % (2 * np.pi)) if a < 0 else (abs(a) %
															 (2 * np.pi))

	def normalize_polar(self, v):
		return np.array([self.normalize_ang(v[0]), min(v[1], self.max_speed)])

	def polar_to_cart(self, v):
		return v[1] * np.array([np.cos(v[0]), np.sin(v[0])])

	# def step(self, action):
	#     action = self.polar_to_cart(self.normalize_polar(action))
	#     if (self.pos + action >= 0).all() and (self.pos + action <
	#                                            1).all():  # stay in bounds
	#       self.pos += action

	#     self.succ = np.linalg.norm(self.pos - self.goal) <= self.goal_dist_thresh
	#     self.crash = np.linalg.norm(self.pos - self.trap) <= self.trap_dist_thresh

	#     self.timestep += 1

	#     obs = self.obs()
	#     r = self.reward_func(self.prev_obs[np.newaxis, :], action[np.newaxis, :],
	#                          obs[np.newaxis, :])[0]
	#     done = self.succ or self.crash
	#     info = {'goal': self.goal, 'succ': self.succ, 'crash': self.crash}

	#     self.prev_obs = obs

	#     return obs, r, done, info

	def step(self, action, old_pos):
		# 0 degrees goes right
		# pi/2 goes up
		# pi goes left
		# 3pi/2 gown down
		action = self.polar_to_cart(self.normalize_polar(action))

		if (old_pos + action >= 0).all() and (old_pos + action <
											   1).all():  # stay in bounds
		  pos = old_pos + action
		else:
		  return old_pos, 0, False, {}

		self.succ = np.linalg.norm(pos - self.goal) <= self.goal_dist_thresh
		self.crash = (np.linalg.norm(pos - self.trap[0]) <= self.trap_dist_thresh) or (np.linalg.norm(pos - self.trap[1]) <= self.trap_dist_thresh)

		self.timestep += 1

		# obs = self.obs()
		# r = self.reward_func(self.prev_obs[np.newaxis, :], action[np.newaxis, :],
		#                      obs[np.newaxis, :])[0]
		r = self.reward_func(old_pos[np.newaxis, :], action[np.newaxis, :],
							 pos[np.newaxis, :])[0]
		done = self.succ or self.crash
		info = {'goal': self.goal, 'succ': self.succ, 'crash': self.crash}

		# self.prev_obs = obs

		return pos, r, done, info

	def reset(self):
		self.pos = np.random.random(2) if self.init_pos is None else deepcopy(
			self.init_pos)
		self.prev_obs = self.obs()
		self.timestep = 0
		return self.prev_obs

	def make_expert_policy(self, noise=0.05, safety_margin=0.01):
		"""Expert goes directly to target, swings around trap if necessary"""

		def policy(obs):
			u = self.goal - obs
			w = self.cart_to_polar(u)
			v = self.trap - obs
			p = v.dot(u)
			# IPython.embed()
			x = obs + u / np.linalg.norm(u) * p
			if p > 0 and np.linalg.norm(
				  v) < self.trap_dist_thresh + safety_margin and np.linalg.norm(
					  x - self.trap) < self.trap_dist_thresh + safety_margin:
				w[0] = self.cart_to_polar(v)[0] + 0.5 * np.pi
			w[0] += np.random.normal(0, 1) * noise
			# IPython.embed()
			return w

		return policy



	# def make_noisy_expert_policy(expert_policy, action_space, eps=0.5):
	  # 	def policy(obs):
			# if np.random.random() < eps:
			#   	return action_space.sample()
			# else:
			#   	return expert_policy(obs)

	  # 	return policy



  

