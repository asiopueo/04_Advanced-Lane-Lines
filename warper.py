import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image
import matplotlib.image as mpimg


src = np.float32([[675,445], [1020,665], [280,665], [605,445]])
dst = np.float32([[1020,0], [1020,665], [280,665], [280,0]])



def imageWarper(img):
	M = cv2.getPerspectiveTransform(src, dst)
	img_size = (img.shape[1], img.shape[0])
	warpedImg = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
	return warpedImg

def imageWarperInv(img):
	M_inv = cv2.getPerspectiveTransform(dst, src)
	img_size = (img.shape[1], img.shape[0])
	dewarpedImg = cv2.warpPerspective(img, M_inv, img_size, flags=cv2.INTER_LINEAR)
	return dewarpedImg


if __name__=='__main__':
	
	image_rgb = mpimg.imread('./test_images/straight_lines1.jpg')

	image_grad_bin = mpimg.imread('./images/gradients_binary.png')
		
	image_warped = imageWarper(image_rgb)

	fig = plt.figure()
	plt.subplot(2,1,1)
	plt.title('Original Image')
	plt.plot(src[:,0], src[:,1], 'ro')
	plt.plot(dst[:,0], src[:,1], 'bo')
	plt.imshow(image_rgb)
	plt.subplot(2,1,2)
	#plt.title('Binary Gradient Image')
	#plt.imshow(image_grad_bin)
	#plt.subplot(3,1,3)
	plt.title('Warped Image')
	plt.imshow(image_warped)

	plt.tight_layout()
	plt.show()

	fig.savefig('./output_images/warped.png')









