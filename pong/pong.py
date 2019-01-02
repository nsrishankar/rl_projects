import numpy as numpy
import random
import gym
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential
from keras.utils import multi_gpu_model
from pong_utils import *
#Environment
env=gym.make("Pong-v0")
#Episode
observation=env.reset()
previous_input=None

#Action Space
UP_ACTION=2 # or 4
DOWN_ACTION=3 # or 5

#Hyperparameters, Variables init
x_train,y_train,reward=[],[],[]
sum_reward=0.
max_episodes=2000
gamma=0.99 # Discounted reward

#Simple Model
model=Sequential()
model.add(Dense(units=150,input_dim=80*80,activation='LeakyReLU'))
model.add(Dense(units=1,activation='sigmoid'))

model.summary()
multi_model=multi_gpu_model(model,gpus=1)
multi_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


#
for this_episode in range(2000):
	current_screen=preprocess(observation)
	x=current_screen-previous_screen if previous_screen is not None else np.zeros(6400) 
	previous_screen=current_screen

	