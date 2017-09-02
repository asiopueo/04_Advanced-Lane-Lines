# Advanced Lane Finding Project

#### The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


---
## Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

<img src="./images/calibration.png" width="600">

The whole code for the camera calibration is contained in the module `calibration.py`.  In order to simplify the code of my project, I have chosen not to integrate this code into `main.py`, but to calculate the distortion coefficients first and paste the result manually into lines 24 to 28 in the main module `main.py`.

The distortion coefficients will be saved as numpy-arrays and are given as follows:

```
mtx =  np.array([ [ 878.36220519,    0.,          641.10477419],
                  [   0.,          850.22573416,  237.43088632],
                  [   0.,            0.,            1.        ]])

dist = np.array([ [-0.24236876,  0.29461093,  0.01910478,  0.00032653, -0.21494586] ])
```



## Pipeline (single images)
#### 1. Provide an example of a distortion-corrected image.
To demonstrate the distortion correction, I have applied the function `undistort()` with the parameters `mtx` and `dist` as they were calculated in the module `calibration.py` to the test image `straight_lines2.jpg`.

The image correction is the second operation of the pipeline in line 35 of `main.py` (right after the conversion RGB->BGR).  Let img_bgr be the imput image with BGR color channels.  Then the undistorted image is returned by the function

```
cv2.undistort(img_bgr, mtx, dist, None, mtx)
```

[Of course, one may also use cv2 instead of mpimg to import images (the former already imports them with BGR-channels).  However, the video images are obviously imported with RGB-channels by moviepy, so the conversion RGB->BGR remains in the code pipeline.]


<img src="./images/undistorted_straight2.png" width="600">

And using the image `test2.jpg`:

<img src="./images/undistorted_test2.png" width="600">

As a step of preprocessing, I have applied this perspective-correction function to all the test images available.



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The calculations of gradients and color transformations are performed in the modules  `gradients.py` and `hls.py`.  These are imported into the main module, `main.py`.  The pipeline in `main.py` can be configured to use all sorts of logical combinations of binary images if gradients and color channels.  In order to figure out appropriate combinations of these channels, I have developed a little helper tool with a GUI which utilizes PyQt4, `sliderAnalyzer.py`:

<img src="./images/sliderAnalyzer.png" width="600">

Using the sliders ("range sliders" with two handles are by default not contained in Qt), one can play around with the thresholds for gradients and color channels.  The text boxes below the image window allow the user to edit logical combinations of the channels.

Identifying the yellow left lane line between seconds 22 and 27 in the project video was one of the trickiest parts.  I chose HLS as color space, and ended up using only the H-channel of all three color channels.  As thresholds, I chose (20,30).  I experienced that the images were surprisingly sensitive with respect to the lower threshold.  I.e. changing the lower threshold from 20 to 25 already made the left lane line disappear.  For the gradients, I only used 'gradMag' with thresholds (65,210).

Hence, using color and gradient thresholds (lines 38 to 55 in `main.py`), I was able to generate a sufficiently clear image:

<img src="./images/gradients_binary.png" width="600">

To summerize, I ended up with
```
  binaryComposite[(H_channel==1) | (gradMag==1)] = 1,
```
although the h-channel was of uttermost importance in this scenario.



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transformations are done in the module `warper.py` by the functions `imageWarper()` and `imageWarperInv()`.  The former function transforms the trapezoid with the corners `src` onto a rectagle with corners `dst`, i.e. warpes the lane into a bird's eye view:

```
src = np.float32([[675,445], [1020,665], [280,665], [605,445]])
dst = np.float32([[1020,0], [1020,665], [280,665], [280,0]])
```
Or in tabular form:

| Source        | Destination       |
|:--------------:|:--------------------:|
| (678,445)     | (1020,0)       	 |
| (1020,665)   | (1020,665)      	 |
| (280,665)     | (280,665)      	 |
| (602,445)     | (280,0)			 |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<img src="./images/warped.png" width="600">



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify the lanes, I performed a perspective transformation like above, but not on a color image, but on a binary image which has already been processed by various gradient and color transforms (cf. **section 2** above).  The warped images then looks as follows:

The identification of lane pixels is done by the function `laneFit()` in the module of the same name.
The pixels are identified by an initial histogram an subsequent sliding window search.  The warped image is divided vertically into a number of strips in which the sliding windows are used to identify the pixels.  After this, all positively identified pixels of each lane (left and right) are glued together again and fitted by a polynomial of order 2.

A second order polynomial is given by

$p(y) = a_2 x^2 + a_1 x + a_0$.

The nonlinear regression is done by a numpy function in lines 177 and 181, respectively:
```
  leftLine.current_fit = np.polyfit(leftLine.ally, leftLine.allx, 2)
```

The coefficients of the polynomial are stored for a few frames (~8 to 12 frames) in a deque and a simple average is calculated (line 41 to 46 in `laneFit.py`; `refresh_avg_fit()`):

$\overline{p}(y) = \overline{a}_2 x^2 + \overline{a}_1 x + \overline{a}_0$,

where

$\overline{a}_i = \frac{1}{N}\sum_{j=1}^N a_{i,j}, \quad i=0,1,2$.

Here is an example of a still test image:

<img src="./images/curvature_test2.png" width="600">





#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radii of curvature of the lane lines are calculated in lines 49 to 65 in the module `laneFit.py`.
I have implemented them as methods `setCurvature()` and `getCurvature()` of the class `Line`.
The method `setCurvature()` calculated the curvature of a lane line and stores the result in a deque.  The method `getCurvature()` is used to retrieve the average value of the deque.

The distance to the center is currently calculated directly in the image pipeline.  This is not a very elegant solution but possible since it is only a one-line command.  It is contained in line 65 of the main module.  The position is calculated as follows:

```
  carPos = (float(rightLane.xbase+leftLane.xbase)-img_bgr.shape[1])/2 * 3.7/700
```

Here, img_bgr is the input image of the pipeline in 'bgr-format'.


Finally, both curvature and position in the lane are printed onto the screen by the function `screenWriter()` in lines 79 to 82 of the main module `main.py`.  This function utilizes the openCV-command `putText()`.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


<img src="./images/laneArea_test3.png" width="600">





---

## Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/llines.mp4).
Please notice that the detected lane in the video has been drawn **white** instead of green as a last-minute decision.  This was due some experimentation with the function `get_colored_warp()` in the module `laneFit.py`.

---

## Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

One of my achievements apart from lane detection itself, is establishing a pipeline structure which enables me to experiment more easily with the parameter values.  This is done by using the 'functions are objects'-paradigm of Python.

Identifying useful thresholds for the gradients and color channels was especially tricky.  I am not very good in tinkering around in a python strip and instead made an attempt to create a small program with PyQt to solve this problem (I later learned that it is also possible to create something similar within a Jupyter notebook).  This has simplified my task profoundly, and I plan to apply the methods onto the more challenging project videos as soon as I have more time.

I have followed the advise given at the end of the lectures and implemented stabilization techniques by utilizing the module `deque`:
```
  from collections import deque
```
I let the length of the deque be determined by the variable `BUFFER_LENGTH` and store the three float-type parameters of the polynomial fit.  When the deque is filled up (i.e. after BUFFER_LENGTH frames), I average them and am able to compare the mean with the parameters which were calculated in the most current frame.  A point of discussion when it comes to these sort of problems is choosing the right metric.  I have chosen a very simple approach by comparing the constant coefficients of the polynomial fits, i.e.

$\mid\overline{a}_0-a_0\mid$.

Of course, when aiming for more stable code, I'd aim for an approach which also involves the leading coefficients of the polynomial fit and a more mathematical approach.

In the same manner I have calculated the mean of the position of the vehicle with respect to the left lane line, and the curvature of the road.  As a rule of thumb, deque lengths between 8 and 12 provided good values.
