import numpy as np
import cv2
import PIL
import matplotlib.image as mpimg
import matplotlib.pyplot as plt





warped_gray = cv2.cvtColor(cv2.imread('./warped.png'), cv2.COLOR_RGB2GRAY)
threshold = (200, 255)
warped_binary = np.zeros_like(warped_gray)
warped_binary[(warped_gray > threshold[0]) & (warped_gray <= threshold[1])] = 1

plt.imshow(warped_binary, cmap='gray')
plt.show()

# OpenCV uses both mathematical and image coordinate systems!
histogram = np.sum(warped_binary[warped_binary.shape[0]//2:,:], axis=0)

plt.plot(histogram)
plt.show()


#out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255









