import sys
import getopt

import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

from moviepy.editor import VideoFileClip
from saver import image_saver, video_saver


from gradients import *
from hls import *
from warper import *
from laneFit import *



leftLane = Line()
rightLane = Line()

mtx =  np.array([ [ 878.36220519,    0.,          641.10477419],
                  [   0.,          850.22573416,  237.43088632],
                  [   0.,            0.,            1.        ]])

dist = np.array([ [-0.24236876,  0.29461093,  0.01910478,  0.00032653, -0.21494586] ])




def pipeline(img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.undistort(img_bgr, mtx, dist, None, mtx)  # Camera correction
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #binaryGradx = absSobelThresh(img, orient='x', sobel_kernel=ksize, thresh=(0,20))
    #binaryGrady = absSobelThresh(img, orient='y', sobel_kernel=ksize, thresh=(0,20))
    binaryMag = magThresh(img_bgr, sobel_kernel=3, thresh=(65,210) )
    #binaryDir = dirThresh(img, sobel_kernel=ksize, thresh=(-np.pi/4.,np.pi/4.) )
    
    binaryGrad = np.zeros_like(gray)
    #pipeline.binaryGrad[((binaryGradx==0)|(binaryGrady==0)) & ((binaryMag==0)|(binaryDir==0)) ] = 1
    binaryGrad[binaryMag==1] = 1

    ##  Color Channels
    h_binary = HLS_Channel(img_bgr, thresh=(20,30), channel='h')
    #l_binary = HLS_Channel(img, 'l', (5,100))
    #s_binary = HLS_Channel(img, 's', (0,100))
    binaryColor = np.zeros_like(gray)
    binaryColor[h_binary==1] = 1
    
    binaryComposite = np.zeros_like(gray)
    binaryComposite[(binaryGrad==1)|(binaryColor==1)] = 1
    
    binaryWarped = imageWarper(binaryComposite)
    fittedWarped = laneFit(binaryWarped, leftLane, rightLane)
    
    fittedWindshield = imageWarperInv(fittedWarped)
    weightedImg = fittedWindshield
    weightedImg = weighted_img(fittedWindshield, img)

    # Improve this measurement
    carPos = float(rightLane.xbase-leftLane.xbase)*3.7/1400 #/2 * 3.7/700
    
    weightedImg = screenWriter(weightedImg, textString = 'Curvature radius of left lane line: ' + '{:.1f}'.format(leftLane.getCurvature()) + 'm', pos=(100,100))
    weightedImg = screenWriter(weightedImg, textString = 'Curvature radius of right lane line: ' + '{:.1f}'.format(rightLane.getCurvature()) + 'm', pos=(100,130))
    weightedImg = screenWriter(weightedImg, textString = 'Position of Vehicle: ' + '{:.2f}'.format(carPos) + 'm', pos=(100,160))

    return weightedImg



def weighted_img(img, initial_img, α=1., β=.2, λ=0.):
    img = np.dstack((img,img,img))
    return cv2.addWeighted(initial_img, α, img, β, λ)

def screenWriter(img, textString, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    textedImg = cv2.putText(img, text=textString, org=pos, fontFace=font, fontScale=.8, color=(0,0,0), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
    return textedImg



def imageProcessing():
    imageRGB = mpimg.imread('./test_images/test1.jpg')
    outputImage = pipeline(imageRGB)

    height, width = (2, 1)
    fig = plt.figure()
    plt.subplot(height, width, 1)
    plt.imshow(imageRGB)
    plt.title('Input of the Pipeline')
    plt.subplot(height, width, 2)
    #plt.imshow(outputImage, cmap='gray')
    plt.imshow(outputImage)
    plt.title('Output of the Pipeline')
    plt.tight_layout()
    plt.show()  
    #image_saver(outputImage)

def videoProcessing():
    clip = VideoFileClip('./test_videos/project_video.mp4').subclip(19,36)
    output_stream = clip.fl_image(pipeline)
    video_saver(output_stream)



def usage():
    print("How to use this program:")
    pass

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'vi', ['Image=', 'Video='])
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
        else:
            usage()
            sys.exit()


if __name__=='__main__':

    main(sys.argv[1:])
    sys.exit()




















