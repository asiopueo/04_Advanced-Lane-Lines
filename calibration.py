import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import PIL




img = cv2.imread('./camera_cal/calibration7.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


nx, ny = 9, 6

ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret==True:
	print('Chessboard pattern recognized!')
	cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
	plt.imshow(img)



objpoints = []
imgpoints = []

objp = np.zeros( (6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) 

imgpoints.append(corners)
objpoints.append(objp)



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#print(mtx)
dst = cv2.undistort(img, mtx, dist, None, mtx)


plt.subplot(2,1,1)
plt.imshow(img)
plt.subplot(2,1,2)
plt.imshow(dst)
plt.show()









