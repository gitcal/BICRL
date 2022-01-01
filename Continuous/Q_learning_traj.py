import numpy as np
import IPython
from itertools import product
import tensorflow as tf
import gym
import os
import datetime
from statistics import mean
from gym import wrappers
from tensorflow.keras import optimizers
import plot_grid

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gym # for environment
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam # adaptive momentum 
from keras.optimizers import SGD # adaptive momentum
import random
import mdp_worlds
import keras
from random import sample
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
class DQLAgent(): 
	
	def __init__(self, env):
		# parameters and hyperparameters
		
		# this part is for neural network or build_model()
		self.state_size = env.state_size # this is for input of neural network node size
		self.action_size = env.action_size # this is for out of neural network node size
		self.env = env
		# this part is for replay()
		self.gamma = 0.95
		self.learning_rate = 0.001
		
		# this part is for adaptiveEGreedy()
		self.epsilon = 1 # initial exploration rate
		self.epsilon_decay = 0.9995
		self.epsilon_min = 0.05
		
		self.memory = deque(maxlen = 2000) # a list with 1000 memory, if it becomes full first inputs will be deleted
		
		self.model = self.build_model()
		self.model_target = self.build_model() #Second (target) neural network
		self.update_target_from_model() #Update weights
		
	def build_model(self):
		# neural network for deep Q learning
		model = Sequential()
		model.add(Dense(100, input_dim = self.state_size, activation = 'relu')) # first hidden layer
		model.add(Dense(100, input_dim = 100, activation = 'relu')) # first hidden layer
		# model.add(Dense(50, input_dim = 50, activation = 'relu')) # first hidden layer
		# model.add(Dense(60, input_dim = 60, activation = 'relu')) # first hidden layer
		model.add(Dense(self.action_size, activation = 'linear')) # output layer
		model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
		return model


	def update_target_from_model(self):
		#Update the target model from the base model
		self.model_target.set_weights(self.model.get_weights())

	def remember(self, state, action, reward, next_state, done):
		# storage
		self.memory.append((state, action, reward, next_state, done))
	
	def act(self, state):
		# acting, exploit or explore
		if random.uniform(0,1) <= self.epsilon:
			# IPython.embed()
			ind = random.sample(list(range(self.env.action_size)),1)[0]
			return env.num_action_grid_map[ind]
		else:
			act_values = self.model.predict(state)
			ind = np.argmax(act_values[0])
			return env.num_action_grid_map[ind]
			
	
	def replay(self, batch_size):
		# training
		
		if len(self.memory) < batch_size:
			return # memory is still not full
		
		minibatch = np.array(random.sample(self.memory, batch_size)) # take 16 (batch_size) random samples from memory

		states = np.concatenate(minibatch[:,0])
		actions = np.concatenate(minibatch[:,1])
		rewards = minibatch[:,2]
		next_states = np.concatenate(minibatch[:,3])
		dones = minibatch[:,4]
		train_target = self.model.predict(states) #Here is the speedup! I can predict on the ENTIRE batch
		next_state_target = self.model.predict(next_states)
		next_state_predict_target = self.model_target.predict(next_states) #Predict from the TARGET
		acts_target = np.argmax(next_state_target, axis=1)
		for i in range(acts_target.shape[0]):
			# next_state_predict[i, acts_target[i]] = next_state_target[i, acts_target[i]]
			# This is simple Q-learning/For double Q-leanring use next_state_predict_target
			temp_target = rewards[i] + (np.logical_not(dones[i]) * 1) * self.gamma * next_state_target[i, acts_target[i]]
			# print(temp_target)
			train_target[i, self.env.action_grid_map[tuple(actions[i])]] = temp_target
		# IPython.embed()
		# train_target = np.expand_dims(rewards, 1) + np.multiply(np.expand_dims(np.logical_not(dones) * 1, 1), self.gamma * next_state_predict_target)
		# index = 0
		self.model.fit(states, train_target, epochs = 1, verbose = 0) # verbose: dont show loss and epoch
		
		# # ORIGINAL
		# minibatch = random.sample(self.memory, batch_size)
		# for state, action, reward, next_state, done in minibatch:
		# 	# if done:
		# 	# 	print(reward)
		# 	if done: # if the game is over, I dont have next state, I just have reward 
		# 		target = reward
		# 	else:
		# 		# print(reward)
		# 		model_tar = self.model.predict(next_state)[0]
		# 		target_model_tar = self.model_target.predict(next_state)[0]
		# 		model_act = np.argmax(self.model.predict(next_state)[0])
		# 		# IPython.embed()
		# 		# target = reward + self.gamma * target_model_tar[model_act] 
		# 		target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])# original
		# 		#IPython.embed()
		# 		# target = R(s,a) + gamma * max Q`(s`,a`)
		# 		# target (max Q` value) is output of Neural Network which takes s` as an input 
		# 		# amax(): flatten the lists (make them 1 list) and take max value
		# 	train_target = self.model.predict(state) # s --> NN --> Q(s,a)=train_target
		# 	# IPython.embed()
		# 	for key_temp, val in self.env.num_action_grid_map.items(): 
		# 		if np.array_equal(val, action[0]):
		# 			key_act = key_temp
		# 	# IPython.embed()
		# 	train_target[0][key_act] = target
		# 	# IPython.embed()
		# 	# if reward == -10:
		# 	# 	IPython.embed()
		# 	# print(target)
		# 	# if np.random.rand() < 0.0005:
		# 	# 	temp = np.array([[1,0.5]])
		# 	# 	print('1',self.model.predict(temp))
		# 	# 	temp = np.array([[0,0.5]])
		# 	# 	print('2',self.model.predict(temp))
		# 	self.model.fit(state, train_target, epochs=1,verbose = 0) # verbose: dont show loss and epoch
	
	def adaptiveEGreedy(self):
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
	
	def test_agent(self):
		#Testing
		print('Training complete. Testing started...')
		#TEST Time
		#   In this section we ALWAYS use exploit don't train any more
		self.modeltest = keras.models.load_model('Q_model')
	
		for e_test in range(1):
		    state = self.env.reset()
		    state = np.reshape(state, [1, 2])
		    tot_rewards = 0
		    for t_test in range(210):
		        action = np.argmax(self.modeltest.predict(state))
		        nstate, reward, done, _ = self.env.step(env.num_action_grid_map[action])
		        nstate = np.reshape(nstate, [1, 2])
		        tot_rewards += reward
		        #DON'T STORE ANYTHING DURING TESTING
		        state = nstate
		        print(state)
		        #done: CartPole fell. 
		        #t_test == 209: CartPole stayed upright
		        if done or t_test == 209: 
		            # rewards.append(tot_rewards)
		            # epsilons.append(0) #We are doing full exploit
		            print("episode: {}/{}, score: {}, e: {}"
		                  .format(e_test, 1, tot_rewards, 0))
		            break;		

if __name__ == "__main__":
	
	# initialize gym environment and agent
	# env = gym.make('CartPole-v0')
	env =  mdp_worlds.Cont2D_v2(10,10,16,0)
	# IPython.embed()
	# env.reset()

	agent = DQLAgent(env)
	batch_size = 32
	episodes = 4000
	for e in range(episodes):
		traj = []
		# initialize environment
		state = env.reset()
		state = np.reshape(state, [1,2])
		traj.append(state)

		
		time = 0 # each second I will get reward, because I want to sustain a balance forever
		# cnt = 0
		
		while True:
			
			# act
			action = agent.act(state)
			
			# step
			# IPython.embed()
			next_state, reward, done, _ = env.step(action)
			# print(reward,done)
			if reward == 2:
				print('********************************************************************************')
			next_state = np.reshape(next_state, [1,2])
			# print(next_state, reward, done,)
			# remember / storage
			agent.remember(state, np.array([action]), reward, next_state, done)
			
			# update state
			state = next_state
			# IPython.embed()
			# replay
			traj.append(state)
			agent.replay(batch_size)
			# if np.random.rand() < 0.0001:
			# 	print(agent.epsilon)
			
			# adjust epsilon
			agent.adaptiveEGreedy()
			
			time += 1
			# if time%100==0:
			# 	print(e, agent.model.predict(np.array([[0.2,0.5]])))
			# 	print(e, agent.model.predict(np.array([[0.2,0.8]])))
			if time >= 200 or done:#done:
				print('episode: {}, time: {}'.format(e, time))
				break
		# if e == episodes - 1:			
		# 	print(len(traj))
		# 	plot_grid.plot_cont_2D_v2(traj)
		# if e%200==0:
		# 	agent.model.save('Q_model')
		# 	# IPython.embed()
		agent.update_target_from_model()
	IPython.embed()
	agent.model.save('Q_model')

	IPython.embed()
