import numpy as numpy
import random

# Preprocessing
def preprocess(raw_image):
	# Only important section of the image
	processed_image=raw_image[35:195]
	#Downsample
	processed_image=preprocess_image[::2,::2,0]
	#Remove background colors
	processed_image[processed_image==109]=0
	processed_image[processed_image==144]=0
	#Setting everything to black
	processed_image[processed_image!=0]=1
	processed_image=processed_image.astype(np.float).flatten()

	return processed_image
