import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, debug_duration = 500, debug_bool = False):
        # calibration params
        self.ret = None
        self.mtx = None
        self.dist = None

        # perspective transform
        self.M = None

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
        self.debug_duration = debug_duration
        self.debug_bool = debug_bool

        # final output
        self.result = None

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
            cv2.waitKey(self.debug_duration)

    def perspective_transform(self, img, src, dst, debug = False):

        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
        if debug:
            print("warped")
            cv2.imshow('debug', self.warped)
            cv2.waitKey(self.debug_duration)

    def abs_sobel_thresh(self, img, orient = 'x', thresh_min=10, thresh_max=255, debug = False, sobel_kernel=3):
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
            cv2.waitKey(self.debug_duration)

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(10, 255), debug = False):
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
            cv2.waitKey(self.debug_duration)

    def dir_threshold(self, img, sobel_kernel=5, thresh=(0.6, 1.6), debug = False):
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
            cv2.waitKey(self.debug_duration)

    def gradient_thresh(self, debug = False):
        combined = np.zeros_like(self.dir_binary, dtype=np.uint8)
        combined[((self.gradx == 1) & (self.grady == 1)) | ((self.mag_binary == 1) & (self.dir_binary == 1))] = 1
        self.combined_grad_binary = combined

        if debug:
            print("combined_grad_thresh")
            cv2.imshow('debug', combined*255)
            cv2.waitKey(self.debug_duration)

    def color_thresh (self, image, thresh=(90, 255), debug = False):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]

        w_binary = np.zeros_like(b)
        w_binary[(b > 200) & (g > 200) & (r > 200)] = 1

        # Store the binary image
        self.s_binary = s_binary + w_binary

        if debug:
            # Plot the result
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(self.s_binary, cmap='gray')
            ax2.set_title('Thresholded S', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()
            print("s_binary")
            cv2.imshow('debug', self.s_binary*255)
            cv2.imshow('warped', self.warped)
            cv2.waitKey(0)

    def final_thresh(self, debug=False):
        final_binary = np.zeros_like(self.s_binary, dtype=np.uint8)
        final_binary[(self.s_binary == 1) | (self.combined_grad_binary == 1)] = 1
        self.final_binary = final_binary

        if debug:
            print("final_binary")
            cv2.imshow('debug', final_binary*255)
            cv2.waitKey(self.debug_duration)

    def find_lane_lines(self, debug = False):
        binary_warped = self.final_binary
        self.ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
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

        left_fitx = left_fit[0] * self.ploty ** 2 + left_fit[1] * self.ploty + left_fit[2]
        right_fitx = right_fit[0] * self.ploty ** 2 + right_fit[1] * self.ploty + right_fit[2]
        # store values
        self.left_fitx = left_fitx
        self.left_fit = left_fit
        self.right_fitx = right_fitx
        self.right_fit = right_fit
        if debug:
            # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            # cv2.imshow("debug", out_img)
            # plt.plot(left_fitx, self.ploty, color='yellow')
            # plt.plot(right_fitx, self.ploty, color='yellow')
            # plt.xlim(0, 1280)
            # plt.ylim(720, 0)
            # print("waiting")
            # cv2.waitKey(self.debug_duration)
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, self.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, self.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, self.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, self.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(left_fitx, self.ploty, color='yellow')
            plt.plot(right_fitx, self.ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show(block=False)

    def calc_curvature(self, debug = False):
        y_eval = np.max(self.ploty)
        left_curverad = ((1 + (2 * self.left_fit[0] * y_eval + self.left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.left_fit[0])
        right_curverad = ((1 + (2 * self.right_fit[0] * y_eval + self.right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.right_fit[0])
        if debug:
            print(left_curverad, right_curverad)

    def draw_frame(self, debug = False):
        # Create an image to draw the lines on
        undist = self.undistorted
        warped = self.final_binary
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        color_warp = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = np.linalg.inv(self.M)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        self.result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
        if debug:
            print("final_lanes")
            cv2.imshow("debug", self.result)
            cv2.imwrite('result.jpg', self.result)
            cv2.waitKey(self.debug_duration)

    def process_image(self, image, show_debug = False):
        offset = 200
        src = np.array([[600, 450], [700,450], [1100,719], [200, 719]], np.float32)
        dst = np.array([[offset, 0], [image.shape[1]-offset, 0], [image.shape[1]-offset, image.shape[0]], [offset, image.shape[0]]], np.float32)
        self.undistort_image(image, debug = show_debug)
        self.perspective_transform(self.undistorted, src, dst, debug = show_debug)
        self.abs_sobel_thresh(self.warped, 'x', debug = show_debug)
        self.abs_sobel_thresh(self.warped, 'y', debug = show_debug)
        self.dir_threshold(self.warped, debug = show_debug)
        self.mag_thresh(self.warped, debug = show_debug)
        self.color_thresh(self.warped, debug = show_debug)
        self.final_thresh(debug = show_debug)
        self.find_lane_lines(debug = show_debug)
        self.calc_curvature(debug = show_debug)
        self.draw_frame(debug = show_debug)


    def process_stream(self, path):
        import skvideo.io
        # open stream
        stream = skvideo.io.vread(path)
        cv2.waitKey(500)
        print("got stream")

        for frame in stream:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.process_image(frame, show_debug=False)
            cv2.imshow("out", self.result)
            cv2.waitKey(1)

        stream.release()
        cv2.destroyAllWindows()









