import numpy as np
import IPython
from itertools import product
import tensorflow as tf
import os
import datetime
from statistics import mean
from gym import wrappers
from tensorflow.keras import optimizers
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gym # for environment
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam # adaptive momentum 
import random
import mdp_worlds
from random import sample
import keras
from mdp_utils import sumexp, logsumexp
import plot_grid
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
class DQLAgent_traj(): 
	
	def __init__(self):
		# parameters and hyperparameters
		# self.env = env
		self.gamma = 0.95		
		self.max_traj_len = 100
		self.env =  mdp_worlds.Cont2D_v2()
		# self.n_traj = n_traj
		self.beta = 100

	def is_goal(state, goal_thr):
		return np.linalg.norm(state - env.goal) <= goal_thr

	def act(self, state):
		# acting, exploit or explore
		if random.uniform(0,1) <= self.epsilon:
			# IPython.embed()
			ind = random.sample(list(range(self.env.action_size)),1)[0]
			return env.action_grid_map[ind]
		else:
			act_values = self.model.predict(state)
			ind = np.argmax(act_values[0])
			return env.action_grid_map[ind]
			


	def obtain_boltz_trajectories(self):
		
		model = keras.models.load_model('Q_model_w')
		# state = np.array([[0, 1]])

		agent = DQLAgent_traj()

		
		
		# initialize environment
		
		# IPython.embed()
		traj = []
		for i in range(40):
			state = self.env.reset()
			# state=np.array([[0.3,0.2]])
			# env.pos = state
			current_state = np.reshape(state, [1,2])
			temp_l = []
			temp_l_plot = []
			time = 0 # each second I will get reward, because I want to sustain a balance forever
			# cnt = 0
			boltzman_rational_trajectory = []
			while True:
				
				# act
				# IPython.embed()
				Q_vals = model.predict(current_state)
				# exp_numerators = np.exp(agent.beta * Q_vals[0])
				# boltzman_probability = exp_numerators / np.sum(exp_numerators)
				# boltz_act = np.random.choice(list(range(env.action_size)), p=boltzman_probability)

				log_numerators = agent.beta * np.array(Q_vals)
				boltzman_log_probs = log_numerators - logsumexp(log_numerators[0])
				boltzman_probability = np.exp(boltzman_log_probs)
				boltz_act = np.random.choice(list(range(self.env.action_size)), p=boltzman_probability[0])
				# boltz_act = np.argmax(Q_vals)
				boltzman_rational_trajectory.append((current_state[0], self.env.num_action_grid_map[boltz_act]))
				# step
				next_state, reward, done, _ = self.env.step(self.env.num_action_grid_map[boltz_act])
				# print(next_state, reward, done)
				# IPython.emsbed()
				current_state = np.reshape(next_state, [1,2])

				time += 1
				# print(agent.epsilon)
				if time >= agent.max_traj_len or done:#done:
					# print('time: {}'.format(time))
					# maybe need to add the following line too
					boltzman_rational_trajectory.append((current_state[0], self.env.num_action_grid_map[13]))
					break
			
			
			
			for i in range(len(boltzman_rational_trajectory)):
				temp_l.append(boltzman_rational_trajectory[i])
				temp_l_plot.append(boltzman_rational_trajectory[i][0])
			traj.append(temp_l)
			# plot_grid.plot_cont_2D_v2(temp_l_plot)
		# IPython.embed()
		return traj




if __name__ == "__main__":
	def is_goal(state, goal_thr):
		return np.linalg.norm(state - env.goal) <= goal_thr
	# initialize gym environment and agent
	# env = gym.make('CartPole-v0')
	env =  mdp_worlds.Cont2D_v2()
	model = keras.models.load_model('Q_model_w')
	# state = np.array([[0, 1]])
	# model.predict(state)
	# IPython.embed()
	# env.reset()
	agent = DQLAgent_traj(env, 20, 50)

	# for e in range(agent.n_traj):
	
	# initialize environment
	
	# IPython.embed()
	for i in range(10):
		state = env.reset()
		# state=np.array([[0.3,0.2]])
		# env.pos = state
		current_state = np.reshape(state, [1,2])
		temp_l = []
		time = 0 # each second I will get reward, because I want to sustain a balance forever
		# cnt = 0
		boltzman_rational_trajectory = []
		while True:
			
			# act
			# IPython.embed()
			Q_vals = model.predict(current_state)
			# exp_numerators = np.exp(agent.beta * Q_vals[0])
			# boltzman_probability = exp_numerators / np.sum(exp_numerators)
			# boltz_act = np.random.choice(list(range(env.action_size)), p=boltzman_probability)

			log_numerators = agent.beta * np.array(Q_vals)
			boltzman_log_probs = log_numerators - logsumexp(log_numerators[0])
			boltzman_probability = np.exp(boltzman_log_probs)
			boltz_act = np.random.choice(list(range(env.action_size)), p=boltzman_probability[0])
			# boltz_act = np.argmax(Q_vals)
			boltzman_rational_trajectory.append((current_state[0], env.num_action_grid_map[boltz_act]))
			# step
			next_state, reward, done, _ = env.step(env.num_action_grid_map[boltz_act])
			# print(next_state, reward, done)
			current_state = np.reshape(next_state, [1,2])

			time += 1
			# print(agent.epsilon)
			if time >= agent.max_traj_len or done:#done:
				# print('time: {}'.format(time))
				boltzman_rational_trajectory.append((current_state[0], None))
				break
		
		
		
		for i in range(len(boltzman_rational_trajectory)):
			temp_l.append(boltzman_rational_trajectory[i][0])
		plot_grid.plot_cont_2D_v2(temp_l)
		# IPython.embed()

