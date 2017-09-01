import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

from collections import deque




BUFFER_LENGTH = 8
EPSILON = 200.
NWINDOWS = 9        # Number of sliding windows



# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.detected = False                               # was the line detected in the last iteration?
        self.xbase = None
        self.allx = None
        self.ally = None

        self.fit_buffer = deque(maxlen=BUFFER_LENGTH)
        self.diffs = np.array([0,0,0], dtype='float')       # difference in fit coefficients between last and new fits
        self.avg_fit = np.array([0,0,0], dtype='float')     # polynomial coefficients averaged over the last n iterations
        self.current_fit = np.array([0,0,0])                # polynomial coefficients for the most recent fit
        
        self.curvature_buffer = deque(maxlen=BUFFER_LENGTH)
        self.curvature = None                               # radius of curvature of the line in some units



    # deviation from
    def deviation(self, current):
        pass


    def refresh_avg_fit(self):
        if (np.absolute(self.diffs[2]) < EPSILON) | (len(self.fit_buffer) < BUFFER_LENGTH):
            self.fit_buffer.append(self.current_fit)
            self.avg_fit = sum(self.fit_buffer)/BUFFER_LENGTH
        else:
            self.detected = False


    def setCurvature(self, ploty):
        ploty_max = np.max(ploty)
        ym_per_pix = 30./720    # meters per pixel in y dimension
        xm_per_pix = 3.7/700    # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        curveRad = ((1 + (2*fit_cr[0]*ploty_max*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        self.curvature_buffer.append(curveRad)


    def getCurvature(self):
        self.curvature = sum(self.curvature_buffer)/BUFFER_LENGTH
        return self.curvature









def get_colored_warp(img, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    #a3 = np.array( [[[10,500],[1000,500],[1000,600],[10,600]]], dtype=np.int32 )
    
    # Draw the lane onto the warped blank image
    #return cv2.fillPoly(warp_zero, a3, 255)#np.int_([pts]), (0,255, 0))
    return cv2.fillPoly(warp_zero, np.int_([pts]), 255)



margin = 100    # Width of the windows +/- margin
minpix = 50     # Minimum number of pixels found to recenter window

def laneFit(img, leftLine, rightLine):
    window_height = np.int(img.shape[0]/NWINDOWS)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])


    

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []


    # Step through the windows one by one
    if (leftLine.detected == False | rightLine.detected == False):
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        #out_img = np.dstack((img, img, img))*255

        midpoint = np.int(histogram.shape[0]/2)
        leftLine.xbase = np.argmax(histogram[:midpoint])
        rightLine.xbase = np.argmax(histogram[midpoint:]) + midpoint

        # Current positions to be updated for each window
        leftx_current = leftLine.xbase
        rightx_current = rightLine.xbase


        if leftLine.detected == False:
            for window in range(NWINDOWS):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = img.shape[0] - (window+1)*window_height
                win_y_high = img.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                # Draw the windows on the visualization image
                #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)  
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((win_y_low <= nonzeroy) & (nonzeroy < win_y_high) & (win_xleft_low <= nonzerox) & (nonzerox < win_xleft_high)).nonzero()[0] # [0] or [1] is arbitrary
                # Folgende Zeile müsste äquivalent mit der obigen Zeile sein:
                #good_left_inds = nonzero[0][((win_y_low <= nonzeroy) & (nonzeroy < win_y_high) & (win_xleft_low <= nonzerox) & (nonzerox < win_xleft_high))]
                left_lane_inds.append(good_left_inds)

                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))


        if rightLine.detected == False:
            for window in range(NWINDOWS):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = img.shape[0] - (window+1)*window_height
                win_y_high = img.shape[0] - window*window_height
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                right_lane_inds.append(good_right_inds)

                if len(good_right_inds) > minpix:        
                   rightx_current = np.int(np.mean(nonzerox[good_right_inds]))





    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftLine.allx = nonzerox[left_lane_inds]    # Geht das nicht auch eleganter mit 'transpose'?
    leftLine.ally = nonzeroy[left_lane_inds] 
    rightLine.allx = nonzerox[right_lane_inds]
    rightLine.ally = nonzeroy[right_lane_inds]

    """
    leftPixels = np.transpose(nonzero)[left_lane_inds]
    rightPixels = np.transpose(nonzero)[left_lane_inds]
    """

    # Fit a second order polynomial to each (polynomial regression)
    leftLine.current_fit = np.polyfit(leftLine.ally, leftLine.allx, 2)
    leftLine.diffs = np.subtract(leftLine.avg_fit, leftLine.current_fit)
    leftLine.refresh_avg_fit()

    rightLine.current_fit = np.polyfit(rightLine.ally, rightLine.allx, 2)
    rightLine.diffs = np.subtract(rightLine.avg_fit, rightLine.current_fit)
    rightLine.refresh_avg_fit()

    """
    left_fit = np.polyfit(np.transpose(leftPixels)[0],np.transpose(leftPixels)[1], 2)
    right_fit = np.polyfit(np.transpose(rightPixels)[0],np.transpose(leftPixels)[1], 2)
    """

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    leftLine.plotx = leftLine.avg_fit[0]*ploty**2 + leftLine.avg_fit[1]*ploty + leftLine.avg_fit[2]
    rightLine.plotx = rightLine.avg_fit[0]*ploty**2 + rightLine.avg_fit[1]*ploty + rightLine.avg_fit[2]

    leftLine.setCurvature(ploty)
    rightLine.setCurvature(ploty)
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    """
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    """

    color_warp = get_colored_warp(img, ploty, leftLine.plotx, rightLine.plotx)
    return color_warp
    




if __name__=='__main__':
    imageBGR = cv2.imread('./output_images/warped.png')
    fittedImage = laneFit(image_bgr)

    plt.imshow(fittedImage)
    plt.show()


