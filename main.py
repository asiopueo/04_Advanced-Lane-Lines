import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from gradients import *
from hls import *
from warper import *
from laneFit import *


def pipeline(img):
	##	Calulation of Gradients
	ksize = 3

	binaryGradx = absSobelThresh(img, orient='x', sobel_kernel=ksize, thresh=(0,20))
	binaryGrady = absSobelThresh(img, orient='y', sobel_kernel=ksize, thresh=(0,20))

	binaryMag = magThresh(img, sobel_kernel=ksize, thresh=(10,50) )
	binaryDir = dirThresh(img, sobel_kernel=ksize, thresh=(-np.pi/4.,np.pi/4.) )

	binaryGrad = np.zeros_like(img)
	binaryGrad[((binaryGradx==1)&(binaryGrady==1)) | ((binaryMag==1)&(binaryDir==1)) ] = 1
	

	##	Color Channels
	hChannel = HLS_Channel(img, 'h', (5,20))
	lChannel = HLS_Channel(img, 'l', (5,100))
	sChannel = HLS_Channel(img, 's', (0,100))

	binaryColor = np.zeros_like(img)
	#binaryColor[()|()|()] = 1

	binaryComposite = np.zeros_like(img)
	binaryComposite[(binaryGrad==1)|(binaryColor==1)] = 1

	##  Final binary image
	binaryComposite = np.copy(sChannel)

	##	Import operations which warp the picture into bird's eye perspective here
	
	binaryWarped = imageWarper(binaryComposite)


	fittedWarped = laneFit(binaryWarped)

	fittedWindshield = imageWarperInv(fittedWarped)

	augmentedPicture = np.copy(fittedWindshield)
	return augmentedPicture



def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)



def screenWriter(img):
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(img, text="Lane", org=(100,200), fontFace=font, fontScale=4., color=(0,0,0), thickness=3, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
	return textedImg



if __name__=='__main__':

	imageRGB = mpimg.imread('./test_images/straight_lines1.jpg')
	outputImage = pipeline(imageRGB)
 	
	plt.subplot(2,1,1)
	plt.imshow(imageRGB)
	plt.title('Input of the Pipeline')
	plt.subplot(2,1,2)
	plt.imshow(outputImage, cmap='gray')
	plt.title('Output of the Pipeline')
	plt.show()





## Video processing block
"""
clip = VideoFileClip('./solidWhiteRight.mp4')
#clip = VideoFileClip('./solidYellowLeft.mp4')

output_handel = 'test_video.mp4'

output_stream = clip.fl_image(pipeline)
output_stream.write_videofile(output_handel, audio=False)
"""



##	Ultimately not necessary, as warper takes care of it:
"""
def regionOfInterest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    
    maskedImage = cv2.bitwise_and(img, mask)
    return maskedImage
"""






















