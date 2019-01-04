import numpy as np
import gym
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential
from keras.utils import multi_gpu_model
from pong_utils import *

#Action Space
PADDLE_UP=2 # or 4
PADDLE_DOWN=3 # or 5

#Hyperparameters, Variables init
max_episodes=2000
gamma=0.99 # Discounted reward
episode=0
previous_input=None
X_train,Y_train,rewards=[],[],[]
sum_rewards=0

#Simple Model
def policygradient_model():
	model=Sequential()
	model.add(Dense(units=150,input_dim=80*80,kernel_initializer='glorot_uniform'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dense(units=1,activation='sigmoid',kernel_initializer='RandomNormal'))

	#multi_model=multi_gpu_model(model,gpus=1)
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	return model

if __name__=="__main__":
	#Environment
	env=gym.make("Pong-v0")
	model=policygradient_model()
	observation=env.reset()

	current_input=preprocess(observation)
	
	while True:
		env.render()

		x=current_input-previous_input if previous_input is not None else np.zeros(80*80) 
		action_prob=model.predict(np.expand_dims(x,axis=0))
		action=PADDLE_UP if np.random.uniform()<action_prob else PADDLE_DOWN
		y=action-2

		X_train.append(x)
		Y_train.append(y)
		observation,reward,done,info=env.step(action) # Environment step
				
		rewards.append(reward)
		sum_rewards+=reward
		
		if done: #Episode over
			X=np.vstack(X_train)
			Y=np.vstack(Y_train)
			discount_rewards=discounted_rewards(rewards,gamma)
			print("Episode {}, This Reward {}, Cumulative Reward {}".format(episode,reward,sum_rewards))
			
			model.fit(x=X,y=Y,sample_weight=discount_rewards)
			
			if(episode%100==0):
				# serialize model to JSON
				model_json=model.to_json()
				with open("pong_model_ep_{}.json".format(episode), "w") as json_file:
					json_file.write(model_json)
					# serialize weights to HDF5
				model.save_weights("pong_model_ep_{}.h5".format(episode))
				print("Saved model to disk")
			episode+=1
			# Reinitialize
			X_train,Y_train,rewards=[],[],[]
			observation=env.reset()
			sum_rewards=0
			previous_input=None
