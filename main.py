import sys
import getopt

import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from gradients import *
from hls import *
from warper import *
from laneFit import *

from moviepy.editor import VideoFileClip





def pipeline(img):
	##	Calulation of Gradients
	ksize = 3

	binaryGradx = absSobelThresh(img, orient='x', sobel_kernel=ksize, thresh=(0,20))
	binaryGrady = absSobelThresh(img, orient='y', sobel_kernel=ksize, thresh=(0,20))

	binaryMag = magThresh(img, sobel_kernel=ksize, thresh=(10,50) )
	binaryDir = dirThresh(img, sobel_kernel=ksize, thresh=(-np.pi/4.,np.pi/4.) )

	binaryGrad = np.zeros_like(img)
	binaryGrad[((binaryGradx==0)|(binaryGrady==0)) & ((binaryMag==0)|(binaryDir==0)) ] = 1
	
	##	Color Channels
	pipeline.h_binary = HLS_Channel(img, 'h', (5,20))
	pipeline.l_binary = HLS_Channel(img, 'l', (5,100))
	pipeline.s_binary = HLS_Channel(img, 's', (0,100))

	pipeline.binaryColor = np.zeros_like(img)
	#binaryColor[()|()|()] = 1

	binaryComposite = np.zeros_like(img)
	pipeline.binaryComposite[(binaryGrad==1)|(pipeline.binaryColor==1)] = 1

	##  Final binary image
	#binaryComposite = np.copy(sChannel)
	##	Import operations which warp the picture into bird's eye perspective here
	binaryWarped = imageWarper(binaryComposite)
	fittedWarped = laneFit(binaryWarped)
	fittedWindshield = imageWarperInv(fittedWarped)

	#if _tweak == 1:
	#	return weighted_img(img, fittedWindshield)
	#else:
	#	return weighted_img(img, fittedWindshield)
	
	return weighted_img(img, fittedWindshield)


def weighted_img(img, initial_img, α=0.2, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def screenWriter(img):
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, text="Lane", org=(100,200), fontFace=font, fontScale=4., color=(0,0,0), thickness=3, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
	return textedImg






def imageProcessing():
	pS = processingStack()
	pS.printStuff()

	imageRGB = mpimg.imread('./test_images/straight_lines1.jpg')
	outputImage = pipeline(imageRGB)

	height, width = (2, 1)

	plt.subplot(height, width, 1)
	plt.imshow(imageRGB)
	plt.title('Input of the Pipeline')

	plt.subplot(height, width, height*width)
	plt.imshow(pipeline.s_binary, cmap='gray')
	#plt.imshow(outputImage, cmap='gray')
	plt.title('Output of the Pipeline')
	plt.tight_layout()
	plt.show()	


def videoProcessing():
	#clip = VideoFileClip('./videos/project_video.mp4')
	clip = VideoFileClip('./videos/harder_challenge_video.mp4')

	output_handel = './harder_challenge_output.mp4'

	output_stream = clip.fl_image(pipeline)
	output_stream.write_videofile(output_handel, audio=False)



def usage():
	print("How to use this program:")
	print("Huh?")





def main(argv):
	try:
		opts, args = getopt.getopt(argv, 'vih', ['Image=', 'Video=', 'help'])
	except getopt.GetoptError as err:
		print(err)
		usage()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-i', '--Image'):
			print('Option: ' + '\'' + arg + '\'')
			imageProcessing()
			
		elif opt in ('-v', '--Video'):
			videoProcessing()
			
		elif opt in ('-h', '--help'):
			usage()
			sys.exit()
			
		elif opt == '-c':
			global _tweak
			_tweak = 1

		else:
			usage()
			sys.exit()



if __name__=='__main__':

	main(sys.argv[1:])
	sys.exit()




















