from __future__ import print_function

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import cv2



def nothing(x):
    pass

def loadImages():
    image1 = cv2.imread("./test_images/test1.jpg")
    image2 = cv2.imread("./test_images/test2.jpg")
    image3 = cv2.imread("./test_images/test3.jpg")
    image4 = cv2.imread("./test_images/test4.jpg")
    image5 = cv2.imread("./test_images/test5.jpg")
    image6 = cv2.imread("./test_images/test6.jpg")

    image = np.concatenate((np.concatenate((image1, image2), axis=0),np.concatenate((image3, image4), axis=0)), axis=1)
    image = np.concatenate((np.concatenate((image5, image6), axis=0), image), axis=1)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return gray


def adjust_threshold():
    
    grayImage = loadImages()

    threshold = [0 , 68]
    kernel = 9
    direction = [0.7, 1.3]
    direction_delta = 0.01


    cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)

    cv2.createTrackbar('threshold[0]', 'testWindow', threshold[0], 255, nothing)
    cv2.createTrackbar('threshold[1]', 'testWindow', threshold[1], 255, nothing)
    #cv2.waitKey(0)
        

    while (True):
        threshold[0] = cv2.getTrackbarPos('threshold[0]', 'testWindow')
        threshold[1] = cv2.getTrackbarPos('threshold[1]', 'testWindow')

        binary = np.zeros_like(grayImage)
        binary[(grayImage >= threshold[0]) & (grayImage <= threshold[1])] = 1

        cv2.imshow('testWindow', 255*binary)
        print(threshold, direction, kernel)

        key = cv2.waitKey(1000)

        if key == 27: # ESC
            break


    cv2.destroyAllWindows()







if __name__=='__main__':
    #image = cv2.imread("./test_images/test1.jpg")
    #cv2.imshow('test', image)
    adjust_threshold()












