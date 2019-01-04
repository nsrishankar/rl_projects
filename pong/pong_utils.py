import numpy as np
import random

#Preprocessing
def preprocess(raw_image):
	# Only important section of the image
	processed_image=raw_image[35:195]
	#Downsample
	processed_image=processed_image[::2,::2,0]
	#Remove background colors
	processed_image[processed_image==109]=0
	processed_image[processed_image==144]=0
	#Setting everything to black
	processed_image[processed_image!=0]=1
	processed_image=processed_image.astype(np.float).flatten()
	
	return processed_image

#Discounted Rewards form array of rewards
def discounted_rewards(rewards,gamma,epsilon=1e-6):
	discounted_rewards=np.zeros_like(rewards)
	running_sum_rewards=0.

	for tstep in reversed(range(0,len(rewards))):
		if rewards[tstep]!=0:
			running_sum_rewards=0 # Reset sum if game is over
		running_sum_rewards=running_sum_rewards*gamma+rewards[tstep] # Gamma=discount factor for reward
		discounted_rewards[tstep]=running_sum_rewards
	# Standardize rewards (0,1) for variance issue	
	discounted_rewards-=np.mean(discounted_rewards)
	discounted_rewards/=np.std(discounted_rewards)+epsilon

	return discounted_rewards