import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import PIL




img = cv2.imread('./camera_cal/calibration2.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


nx, ny = 9, 6

ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret==True:
	print('Chessboard pattern recognized!')
	cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
	#plt.imshow(img)


objpoints = []
imgpoints = []

objp = np.zeros( (6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) 

imgpoints.append(corners)
objpoints.append(objp)



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)



img = cv2.imread('./test_images/test2.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

fig = plt.figure()

plt.subplot(2,1,1)
plt.title('Original Image')
plt.imshow(img)

plt.subplot(2,1,2)
plt.title('Undistorted Image')
plt.imshow(dst)

plt.tight_layout()
plt.show()

fig.savefig('./output/undistorted_test2.png')







