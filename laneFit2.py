import numpy as np
import cv2
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as pyplot


histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255









