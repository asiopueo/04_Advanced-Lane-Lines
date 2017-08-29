import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

from collections import deque

BUFFER_LENGTH = 12
EPSILON = 10.
NWINDOWS = 9        # Number of sliding windows



# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.detected = False       # was the line detected in the last iteration?
        self.recent_xfitted = []    # x values of the last n fits of the line
        self.bestx = None           # average x values of the fitted line over the last n iterations
        self.fit_buffer = deque(BUFFER_LENGTH*[], BUFFER_LENGTH)
        self.avg_fit = np.array([0,0,0], dtype='float') # polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])]          # polynomial coefficients for the most recent fit
        self.radius_of_curvature = None                 # radius of curvature of the line in some units
        self.line_base_pos = None                       # distance in meters of vehicle center from the line
        self.diffs = np.array([0,0,0], dtype='float')   # difference in fit coefficients between last and new fits
        self.allx = None                                # x values for detected line pixels
        self.ally = None                                # y values for detected line pixels

        self.average_fit = [np.array([False])]
        #self.average_radius = 


    # deviation from
    def deviation(self, current):
        pass


    def refresh_avg_fit(self):
        if np.sum(self.diffs) < EPSILON:
            self.fit_buffer.appendleft(self.current_fit)
            self.avg_fit = sum(self.fit_buffer)#/BUFFER_LENGTH
        else:
            self.detected = False


    def get_lane_curvature():
        pass


    def get_radii(self, ploty):
        # Calculate the curvature of the lanes:
        y_eval = np.max(ploty)
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        return left_curverad, right_curverad


    # Maybe try to eliminate img in this method
def get_colored_warp(img, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    a3 = np.array( [[[10,500],[1000,500],[1000,600],[10,600]]], dtype=np.int32 )
    
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
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        #print(midpoint)
        #print(leftx_current)
        #print(rightx_current)


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
    leftx = nonzerox[left_lane_inds]    # Geht das nicht auch eleganter mit 'transpose'?
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    """
    leftPixels = np.transpose(nonzero)[left_lane_inds]
    rightPixels = np.transpose(nonzero)[left_lane_inds]
    """

    # Fit a second order polynomial to each (polynomial regression)
    leftLine.current_fit = np.polyfit(lefty, leftx, 2)
    leftLine.diffs = np.subtract(leftLine.avg_fit,leftLine.current_fit)
    leftLine.refresh_avg_fit()

    rightLine.current_fit = np.polyfit(righty, rightx, 2)
    rightLine.diffs = np.subtract(rightLine.avg_fit,rightLine.current_fit)
    rightLine.refresh_avg_fit()

    

    """
    left_fit = np.polyfit(np.transpose(leftPixels)[0],np.transpose(leftPixels)[1], 2)
    right_fit = np.polyfit(np.transpose(rightPixels)[0],np.transpose(leftPixels)[1], 2)
    """

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = leftLine.avg_fit[0]*ploty**2 + leftLine.avg_fit[1]*ploty + leftLine.avg_fit[2]
    right_fitx = rightLine.avg_fit[0]*ploty**2 + rightLine.avg_fit[1]*ploty + rightLine.avg_fit[2]

    
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    """
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    return out_img
    """
    #print(left_fitx)
    #print(right_fitx)

    color_warp = get_colored_warp(img, ploty, left_fitx, right_fitx)
    #plt.imshow(color_warp)
    #plt.show()
    #left_curverad, right_curverad = get_radii(ploty)

    return color_warp
    




if __name__=='__main__':
    imageBGR = cv2.imread('./output_images/warped.png')
    fittedImage = laneFit(image_bgr)

    plt.imshow(fittedImage)
    plt.show()


