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

def pipeline(img):
    ksize = 3

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    

    #binaryGradx = absSobelThresh(img, orient='x', sobel_kernel=ksize, thresh=(0,20))
    #binaryGrady = absSobelThresh(img, orient='y', sobel_kernel=ksize, thresh=(0,20))
    binaryMag = magThresh(img_bgr, sobel_kernel=ksize, thresh=(65,210) )
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
    
    #plt.imshow(binaryColor, cmap='gray')
    #plt.show()
    
    binaryComposite = np.zeros_like(gray)
    binaryComposite[(binaryGrad==1)|(binaryColor==1)] = 1
    
    #pipeline.buffer.appendleft(binaryComposite)
    #binaryComposite = sum(pipeline.buffer)
    ##  Final binary image
    #   binaryComposite = np.copy(sChannel)
    ##  Import operations which warp the picture into bird's eye perspective here
    binaryWarped = imageWarper(binaryComposite)
    fittedWarped = laneFit(binaryWarped, leftLane, rightLane)
    
    #plt.imshow(fittedWarped, cmap='gray')
    #plt.show()    

    fittedWindshield = imageWarperInv(fittedWarped)
    weightedImg = fittedWindshield
    weightedImg = weighted_img(fittedWindshield, img)

    #weightedImg = screenWriter(weightedImg, left_curveRad, right_curveRad)
    #print(left_curveRad, 'm', right_curveRad, 'm')
    return weightedImg




def weighted_img(img, initial_img, α=1., β=.2, λ=0.):
    img = np.dstack((img,img,img))
    #print(img.shape)
    return cv2.addWeighted(initial_img, α, img, β, λ)




def screenWriter(img, left_cur, right_cur):
    font = cv2.FONT_HERSHEY_SIMPLEX
    left_str = 'Curvature radius of left lane line: ' + '{:.1f}'.format(left_cur) + 'm'
    right_str = 'Curvature radius of right lane line: ' + '{:.1f}'.format(right_cur)  + 'm'
    textedImg = cv2.putText(img, text=left_str, org=(100,100), fontFace=font, fontScale=1., color=(0,0,0), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
    textedImg = cv2.putText(textedImg, text=right_str, org=(100,150), fontFace=font, fontScale=1., color=(0,0,0), thickness=2, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
    return textedImg


def imageProcessing():

    imageRGB = mpimg.imread('./test_images/test1.jpg')
    #imageRGB = cv2.imread('./test_images/test1.jpg')
    outputImage = pipeline(imageRGB)

    height, width = (2, 1)

    #fig = plt.figure()
    #plt.subplot(height, width, 1)
    #plt.imshow(imageRGB)
    #plt.title('Input of the Pipeline')

    #plt.subplot(height, width, height*width)
    #plt.imshow(outputImage, cmap='gray')
    plt.imshow(outputImage)
    plt.title('Output of the Pipeline')
    plt.tight_layout()
    plt.show()  
    #image_saver(outputImage)




def videoProcessing():
    clip = VideoFileClip('./test_videos/project_video.mp4')#.subclip(19,27)
    output_stream = clip.fl_image(pipeline)
    video_saver(output_stream)



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


pipeline.flag = 0

if __name__=='__main__':

    main(sys.argv[1:])
    sys.exit()




















