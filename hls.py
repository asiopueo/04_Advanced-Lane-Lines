import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2







def HLS_Channel(img_bgr, channel='l', thresh=(0,255)):

	img_hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
	
	if channel == 'h':
		colorChannel = img_hls[:,:,0]		
	
	elif channel == 'l':
		colorChannel = img_hls[:,:,1]
	
	elif channel == 's':
		colorChannel = img_hls[:,:,2]

	else:
		raise

	binary = np.zeros_like(colorChannel)
	binary[(colorChannel >= thresh[0]) & (colorChannel <= thresh[1])] = 1

	return binary





if __name__=='__main__':

	image_bgr = cv2.imread('./signs_vehicles_xygrad.png')
	
	H_binary = HLS_Channel(image_bgr, 's', thresh=(5,30))
	

	plt.imshow(H_binary, cmap='gray')
	plt.show()












