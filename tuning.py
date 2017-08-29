from __future__ import print_function

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import cv2



def adjust_threshold():
    image1 = cv2.imread("./test_images/test1.jpg")
    image2 = cv2.imread("./test_images/test2.jpg")
    image3 = cv2.imread("./test_images/test3.jpg")
    image4 = cv2.imread("./test_images/test4.jpg")
    image5 = cv2.imread("./test_images/test5.jpg")
    image6 = cv2.imread("./test_images/test6.jpg")

    image = np.concatenate((np.concatenate((image1, image2), axis=0),np.concatenate((image3, image4), axis=0)), axis=1)
    image = np.concatenate((np.concatenate((image5, image6), axis=0), image), axis=1)

    threshold = [0 , 68]

    kernel = 9

    direction = [0.7, 1.3]
    direction_delta = 0.01

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


    cv2.namedWindow('testWindow', cv2.WINDOW_NORMAL)

    cv2.waitKey(0)

    while True:
        key = cv2.waitKey(1000)
        
        #print("key = ", key)

        if key == 65429: # key "Home"
            if threshold[0] > 0:
                threshold[0] = threshold[0] - 1
            if direction[0] > 0:
                direction[0] = direction[0] - direction_delta

        if key == 65434: # key "PgUp"
            if threshold[0] < threshold[1]:
                threshold[0] = threshold[0] + 1

            if direction[0] < direction[1] - direction_delta:
                direction[0] = direction[0] + direction_delta

        if key == 65430: # left arrow
            if threshold[1] > threshold[0]:
                threshold[1] = threshold[1] - 1

            if direction[1] > direction[0] + direction_delta:
                direction[1] = direction[1] - direction_delta

        if key == 65432: # right arrow
            if threshold[1] < 255:
                threshold[1] = threshold[1] + 1

            if direction[1] < np.pi/2:
                direction[1] = direction[1] + direction_delta

        if key == 65436: # key "End"
            if(kernel > 2):
                kernel = kernel - 2
        if key == 65435: # key "PgDn"
            if(kernel < 31):
                kernel = kernel + 2

        if key == 27: # ESC
            break

        
        binary = np.zeros_like(image)
        binary[(gray >= threshold[0]) & (gray <= threshold[1])] = 1

        cv2.imshow('testWindow', 255*binary)
        print(threshold, direction, kernel)




if __name__=='__main__':
    #image = cv2.imread("./test_images/test1.jpg")
    #cv2.imshow('test', image)
    adjust_threshold()




