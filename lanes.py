import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

class lane_line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

class detector:
    def __init__(self):
        # calibration params
        self.ret = None
        self.mtx = None
        self.dist = None

        # frame holders
        self.undistorted = None
        self.warped = None

        # binary images
        self.gradx = None
        self.grady = None
        self.mag_binary = None
        self.combined_grad_binary = None
        self.s_binary = None
        self.final_binary = None
        self.dir_binary = None

        # were lines detected in the last iteration?
        self.detected = False

        # detected line polynomials
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = np.linspace(0, 719, num=720)

    def get_calibration(self, file_path, debug = False):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        # images = glob.glob('../camera_cal/calibration*.jpg')
        images = glob.glob(file_path)

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                if debug:
                    img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

        cv2.destroyAllWindows()
        self.ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(self.ret)

    def undistort_image(self, image, debug = False):
        # if self.ret:
        self.undistorted = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        if debug:
            print("undistorted")
            cv2.imshow('debug', self.undistorted)
            cv2.waitKey(0)

    def perspective_transform(self, img, src, dst, debug = False):

        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(src, dst)
        self.warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
        if debug:
            print("warped")
            cv2.imshow('debug', self.warped)
            cv2.waitKey(0)

    def abs_sobel_thresh(self, img, orient = 'x', thresh_min=20, thresh_max=255, debug = False, sobel_kernel=3):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
             abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
             abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        #scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(abs_sobel, dtype=np.uint8)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(abs_sobel >= thresh_min) & (abs_sobel <= thresh_max)] = 1

        # Store the result
        if orient == 'x':
            self.gradx = np.copy(binary_output)
        else:
            self.grady = np.copy(binary_output)

        if debug:
            print("abs_sobel")
            cv2.imshow('debug', binary_output*255)
            cv2.waitKey(0)

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(20, 255), debug = False):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag, dtype=np.uint8)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        # Store the binary image
        self.mag_binary = binary_output

        if debug:
            print("mag_thresh")
            cv2.imshow('debug', binary_output*255)
            cv2.waitKey(0)

    def dir_threshold(self, img, sobel_kernel=5, thresh=(0.7, 1.3), debug = False):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir, dtype=np.uint8)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Store the binary image
        self.dir_binary = binary_output

        if debug:
            print("dir_thresh")
            cv2.imshow('debug', binary_output*255)
            cv2.waitKey(0)

    def gradient_thresh(self, debug = False):
        combined = np.zeros_like(self.dir_binary, dtype=np.uint8)
        combined[((self.gradx == 1) & (self.grady == 1)) | ((self.mag_binary == 1) & (self.dir_binary == 1))] = 1
        self.combined_grad_binary = combined

        if debug:
            print("combined_grad_thresh")
            cv2.imshow('debug', combined*255)
            cv2.waitKey(0)

    def color_thresh (self, image, s_thresh_min = 170, s_thresh_max = 255, debug = False):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel, dtype=np.uint8)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        # Store the binary image
        self.s_binary = s_binary

        if debug:
            print("s_binary", s_binary.shape)
            cv2.imshow('debug', s_binary*255)
            cv2.waitKey(0)

    def final_thresh(self, debug=False):
        final_binary = np.zeros_like(self.s_binary, dtype=np.uint8)
        final_binary[(self.s_binary == 1) | (self.combined_grad_binary == 1)] = 1
        self.final_binary = final_binary

        if debug:
            print("final_binary")
            cv2.imshow('debug', final_binary*255)
            cv2.waitKey(0)

    # def window_mask(self, width, height, img_ref, center, level):
    #     output = np.zeros_like(img_ref)
    #     output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    #     max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    #     return output
    #
    # def find_window_centroids(self, warped, window_width, window_height, margin):
    #
    #     window_centroids = []  # Store the (left,right) window centroid positions per level
    #     window = np.ones(window_width)  # Create our window template that we will use for convolutions
    #
    #     # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    #     # and then np.convolve the vertical image slice with the window template
    #
    #     # Sum quarter bottom of image to get slice, could use a different ratio
    #     l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
    #     l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    #     r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
    #     r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)
    #
    #     # Add what we found for the first layer
    #     window_centroids.append((l_center, r_center))
    #
    #     # Go through each layer looking for max pixel locations
    #     for level in range(1, (int)(warped.shape[0] / window_height)):
    #         # convolve the window into the vertical slice of the image
    #         image_layer = np.sum(
    #             warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
    #             :], axis=0)
    #         conv_signal = np.convolve(window, image_layer)
    #         # Find the best left centroid by using past left center as a reference
    #         # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
    #         offset = window_width / 2
    #         l_min_index = int(max(l_center + offset - margin, 0))
    #         l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
    #         l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
    #         # Find the best right centroid by using past right center as a reference
    #         r_min_index = int(max(r_center + offset - margin, 0))
    #         r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
    #         r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
    #         # Add what we found for that layer
    #         window_centroids.append((l_center, r_center))
    #
    #     return window_centroids

    def find_lane_lines(self):
        binary_warped = self.final_binary
        if not self.detected:
            # Assuming you have created a warped binary image called "binary_warped"
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
            # Create an output image to draw on and  visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] / 2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Choose the number of sliding windows
            nwindows = 9
            # Set height of windows
            window_height = np.int(binary_warped.shape[0] / nwindows)
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Current positions to be updated for each window
            leftx_current = leftx_base
            rightx_current = rightx_base
            # Set the width of the windows +/- margin
            margin = 100
            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []
            self.detected = True

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

        else:
            left_fit = self.left_fit
            right_fit = self.right_fit
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            margin = 100
            left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
            right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        left_fitx = left_fit[0] * self.ploty ** 2 + left_fit[1] * self.ploty + left_fit[2]
        right_fitx = right_fit[0] * self.ploty ** 2 + right_fit[1] * self.ploty + right_fit[2]
        # store values
        self.left_fitx = left_fitx
        self.left_fit = left_fit
        self.right_fitx = right_fitx
        self.right_fit = right_fit

    def calc_curvature(self, debug = False):
        y_eval = np.max(self.ploty)
        left_curverad = ((1 + (2 * self.left_fit[0] * y_eval + self.left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.left_fit[0])
        right_curverad = ((1 + (2 * self.right_fit[0] * y_eval + self.right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.right_fit[0])
        if debug:
            print(left_curverad, right_curverad)

    def draw_frame(self, debug = False):
        # Create an image to draw the lines on
        undist = self.undistorted
        warped = self.warped
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = self.mtx.inverse()
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        plt.imshow(result)

    def process_image(self, image):
        db = True
        offset = 200
        img_shape = (image.shape[1], image.shape[0])
        src = np.array([[600, 450], [700,450], [1100,719], [200, 719]], np.float32)
        dst = np.array([[offset, 0], [image.shape[1]-offset, 0], [image.shape[1]-offset, image.shape[0]], [offset, image.shape[0]]], np.float32)
        self.undistort_image(image, debug = db)
        self.perspective_transform(self.undistorted, src, dst, debug = db)
        self.abs_sobel_thresh(self.warped, 'x', debug = db)
        self.abs_sobel_thresh(self.warped, 'y', debug = db)
        self.dir_threshold(self.warped, debug = db)
        self.mag_thresh(self.warped, debug = db)
        self.color_thresh(self.warped, debug = db)
        self.final_thresh(debug = db)
        self.find_lane_lines()
        self.calc_curvature(debug = db)
        self.draw_frame()




