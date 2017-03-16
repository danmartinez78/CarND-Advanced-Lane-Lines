

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/original.jpg "Original"
[image2]: ./output_images/undistorted.jpg "Undistorted"
[image3]: ./output_images/undistorted_lanes.jpg "Undistorted Example"
[image4]: ./output_images/original_lanes.jpg "Original Lanes w/ distortion removed"
[image5]: ./output_images/binary_lanes.jpg "Binary"
[image6]: ./output_images/warped_lanes.jpg "Perspctive Shift"
[image7]: ./examples/color_fit_lines.jpg "Fit Visual"
[image8]: ./output_images/final_lanes.jpg "Output"
[video1]: ./result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. The follwing documentation will cover the required points in the rubric


###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in a member function of my detector class (lines 31-64 in lanes.py)  

The member function takes two arguments, a string variable containing the file path to the requisite calibration images and a debug flag. The code iterates through the calibration checkerboard images one at a time. For each image, the (x, y, z) coordinates of the chessboard corners in the world frame are detected, assuming a z value of 0. `imgpoints` is then appended with corner coordinates when all chessboard corners in a test image have been successfully detected. `objpts` is appended with the local frame values of the corner locations, which should be the same for all given test images of the same chess board.    

After iterating through all the test images `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  The distortion correction can be applied to a test image using the `cv2.undistort()` function with a sample result shown here: 

![alt text][image1]
![alt text][image2]

The first image shows the checkerboard with no distortion correction applied. The second image shows the effect of applying the distortion corrections that we computed.

The debug flag in the member function, enables or disables a debug output that shows the corner detections on each calibration image as they are process.

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
The following image shows the correction applied to image `test4.jpg`. The distortion in the image is greatly reduced. This is the first step in my image processing pipeline and produces fairly good distortion to further process. 

![alt text][image4]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used only color thresholding to generate my binary images. The process is performed by a member function (`color_thresh` lines 76 through 107 in `lanes.py`).  Here's an example of my output for this step. For this example, the color thresholdng was performed on the undistorted image from part 1 for easy comparison. I utilized the HSV colorspace in order to segment out both white and yellow areas in the image individually, and then combined them into one color thresholded image as seen below.

![alt text][image5]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

My perspective transform is performed by the member function `perspective_transform` (lines 70-74 in `lanes.py`).  The function takes an input image and arrays for both source (`src`) and destination (`dst`) points.  I utilized mathplotlib in order to generate an interactive figure that displayed selected pixel location coordinates in order to determine the best src points (`point_picker.py`). The results were very close to the example points and so I decided to hardcode those values:

``` src = np.array([[585, 460], [203, 720], [1127,720], [695, 460]], np.float32)```

``` dst = np.array([[320, 0], [320, 720], [900,720], [900, 0]], np.float32)```

The following image shows the perspective transform applied to the binary lane image. Both yellow and white lane markings are detected and seen from a more orthagonal perspective. This is vital for the next steps in my pipeline.

![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane line pixels were located using a sliding window search. First a histogram of the lower section of the transformed binary image was constructed. The peaks in this histogram identified the pixel locations (on the x axis) where the lane lines were likely to exist. Using that information, a sliding window search was performed to locate the rest of the lane line pixels. This is all performed in the member function `find_lane_lines`(lines 109 to 261 in `lanes.py`). After the lane lines are detected for the first time, this information is utilized to perform a more narrow search. Having found the (x,y) values for the pixels corresponding to the left and right lane markings, a 2nd order polynomial is generated to best fit the points, giving us the equation for each lane line. Then, some averaging is done in order to smooth the transition between frames. The averaging is performed by storing the polyfit objects for each lane line in it's own member value and then performing an exponential moving average every frame. This required some tuning, but I saw good success with an alpha value of 0.1

Additionally, this function takes a debug argument that enables a matplotlib figure displaying the binary warped image overlayed with lane lines and search windows. 

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is also computed in the member function `find_lane_lines`(lines 109 to 261 in `lanes.py`). 

```     y_eval = np.max(self.ploty)
        midx = binary_warped.shape[1]/2.
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty * ym_per_pix, self.left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty * ym_per_pix, self.right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        pos = ((self.left_fitx[-1] + self.right_fitx[-1]) / 2. - midx) * xm_per_pix
```

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 263 through 279 in my code in `lanes.py` in the member function `draw_frame()`.  Here is an example of my result on a test image:

![alt text][image8]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach to this project was to create a lane detection class that stored all requisite values in member variables and performed all frame processing steps in member functions. Initially, my member functions only took parameter arguments and returned no images. The images to be processed were loaded from member variables and the result was stored in another corresponding member variable. The idea being, once instantiated and calibrated, it would require only one function call to process a single frame. This became very difficult to manage in practice. 

I fixed this by having all my member functions take in an image argument and return the result. I added a new member function in order to process a single image frame and provide all debug output as necessary. The `process_image(self, image, debug_flag=False)` member function then calls and stores the results of each portion of the pipeline and returns the final mapped lane image as a result. This proved to be much more manageable.

The `process_stream(self, path, save = False, debug_flag = False)` function then only needs to open the video stream, and call `process_image()` once for each frame. It then writes or displays the resulting frame as directed by the corresponding flags.

In creating my pipeline, I found that only thresholding the s-channel of the HSL colorspace for each frame did not provide good results for both white and yellow lane markings. I decided to use the same strategy as I did for project 1 and threshold HSV for both white and yellow and combine the results. 

Additionally, I found that gradient thresholding did not necessarily improve the line detection and so, I removed that.
 
Smoothing my line transitions from frame to frame, helped quite a bit. In addition, adding checks for too few line points allowed the algorithm to proceed in places where the lane markings were not clear due to lighting effects in the first challenge video.