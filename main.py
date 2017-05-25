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

	pipeline.binaryGrad = np.zeros_like(img)
	pipeline.binaryGrad[((binaryGradx==0)|(binaryGrady==0)) & ((binaryMag==0)|(binaryDir==0)) ] = 1
	
	##	Color Channels
	pipeline.h_binary = HLS_Channel(img, 'h', (5,20))
	pipeline.l_binary = HLS_Channel(img, 'l', (5,100))
	pipeline.s_binary = HLS_Channel(img, 's', (0,100))

	pipeline.binaryColor = np.zeros_like(img)
	#binaryColor[()|()|()] = 1

	pipeline.binaryComposite = np.zeros_like(img)
	pipeline.binaryComposite[(pipeline.binaryGrad==1)|(pipeline.binaryColor==1)] = 1

	##  Final binary image
	#	binaryComposite = np.copy(sChannel)
	##	Import operations which warp the picture into bird's eye perspective here
	pipeline.binaryWarped = imageWarper(pipeline.binaryComposite)
	pipeline.fittedWarped, left_curveRad, right_curveRad = laneFit(pipeline.binaryWarped)
	pipeline.fittedWindshield = imageWarperInv(pipeline.fittedWarped)
	pipeline.weightedImg = weighted_img(img, pipeline.fittedWindshield)
	pipeline.weightedImg = screenWriter(pipeline.weightedImg, left_curveRad, right_curveRad)

	print(left_curveRad, 'm', right_curveRad, 'm')
	
	return pipeline.weightedImg


def weighted_img(img, initial_img, α=0.2, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


def screenWriter(img, left_cur, right_cur):
	font = cv2.FONT_HERSHEY_SIMPLEX
	left_str = 'Curvature radius of left lane line: ' + '{:.1f}'.format(left_cur) + 'm'
	right_str = 'Curvature radius of right lane line: ' + '{:.1f}'.format(right_cur)  + 'm'
	textedImg = cv2.putText(img, text=left_str, org=(100,100), fontFace=font, fontScale=1., color=(0,0,0), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
	textedImg = cv2.putText(textedImg, text=right_str, org=(100,150), fontFace=font, fontScale=1., color=(0,0,0), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
	return textedImg






def imageProcessing():

	imageRGB = mpimg.imread('./test_images/test2.jpg')
	outputImage = pipeline(imageRGB)

	height, width = (2, 1)

	fig = plt.figure()
	plt.subplot(height, width, 1)
	plt.imshow(imageRGB)
	plt.title('Input of the Pipeline')

	plt.subplot(height, width, height*width)
	plt.imshow(outputImage, cmap='gray')
	#plt.imshow(outputImage, cmap='gray')
	plt.title('Output of the Pipeline')
	plt.tight_layout()
	plt.show()	

	fig.savefig('./output/curvature_test2.png')




def videoProcessing():
	#clip = VideoFileClip('./videos/project_video.mp4')
	clip = VideoFileClip('./videos/harder_challenge_video.mp4')

	output_handel = './harder_challenge_output.mp4'

	output_stream = clip.fl_image(pipeline)
	output_stream.write_videofile(output_handel, audio=False)



def usage():
	print("How to use this program:")
	pass





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




















