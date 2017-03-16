import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

class detector:
    def __init__(self, debug_duration = 500, debug_bool = False):
        # calibration params
        self.ret = None
        self.mtx = None
        self.dist = None

        # perspective transform
        self.M = None

        # were lines detected in the last iteration?
        self.detected = False

        # detected line params
        self.left_fit = None
        self.left_fitx = None
        self.right_fit = None
        self.right_fitx = None
        self.ploty = None
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

    def undistort_image(self, image):
        # if self.ret:
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    def perspective_transform(self, image, src, dst):
        # Compute and apply perspective transform
        img_size = (image.shape[1], image.shape[0])
        self.M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, self.M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    def color_thresh(self, image, h_thresh=(15, 45), s_thresh=(90, 255), v_thresh=(90, 255)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # segment yellow lane markings
        yellow_binary = np.zeros_like(h_channel)
        h = np.zeros_like(h_channel)
        s = np.zeros_like(s_channel)
        v = np.zeros_like(v_channel)

        h[(h_channel > h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
        s[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        v[(v_channel > v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

        yellow_binary[(h==1)&(s==1)&(v==1)] = 1

        # segment white lane markings
        white_binary = np.zeros_like(h_channel)
        h = np.zeros_like(h_channel)
        s = np.zeros_like(s_channel)
        v = np.zeros_like(v_channel)

        h[(h_channel > 0) & (h_channel <= 255)] = 1
        s[(s_channel > 0) & (s_channel <= 45)] = 1
        v[(v_channel > 210) & (v_channel <= 255)] = 1

        white_binary[(h == 1) & (s == 1) & (v == 1)] = 1

        # Combine and store the binary image
        return np.add(yellow_binary, white_binary)

    def find_lane_lines(self, binary_warped, debug = False):
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
            self.detected = True

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

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        if len(leftx) > 2000:
            left_fit = np.polyfit(lefty, leftx, 2)
        else:
            left_fit = self.left_fit

        if len(rightx) > 2000:
            right_fit = np.polyfit(righty, rightx, 2)
        else:
            right_fit = self.right_fit

        if self.left_fit != None:
            # sanity checks needed here
            alpha = 0.2
            self.left_fit = alpha * left_fit + (1. - alpha) * self.left_fit
            self.right_fit = alpha * right_fit + (1. - alpha) * self.right_fit
        else:
            self.left_fit = left_fit
            self.right_fit = right_fit

        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

        if debug:
            margin = 100
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx - margin, self.ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx + margin, self.ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx - margin, self.ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx + margin, self.ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            plt.clf()
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            plt.imshow(result)
            plt.plot(self.left_fitx, self.ploty, color='yellow')
            plt.plot(self.right_fitx, self.ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.show(block = False)
            plt.pause(0.05)

        y_eval = np.max(self.ploty)
        midx = binary_warped.shape[1]/2.
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.ploty * ym_per_pix, self.left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty * ym_per_pix, self.right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        pos = ((self.left_fitx[-1] + self.right_fitx[-1]) / 2. - midx) * xm_per_pix

        return left_curverad, right_curverad, pos

    def draw_frame(self, undist, warped):
        # Create an image to draw the lines on
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
        return cv2.addWeighted(undist, 1, newwarp, 0.5, 0)

    def process_image(self, image, debug_flag=False):
        src = np.array([[585, 460], [203, 720], [1127,720], [695, 460]], np.float32)
        dst = np.array([[320, 0], [320, 720], [900,720], [900, 0]], np.float32)
        undistorted = self.undistort_image(image)
        warped = self.perspective_transform(undistorted, src, dst)
        binary_warped = self.color_thresh(warped, h_thresh=(10, 80), s_thresh=(75, 255), v_thresh=(75, 255))
        left_curverad, right_curverad, pos = self.find_lane_lines(binary_warped, debug=debug_flag)
        avg_radius = (left_curverad + right_curverad)/2.
        output = self.draw_frame(undistorted, binary_warped)

        # write radius of curvature and center_offset
        cv2.putText(output, 'Radius of Curvature: %.2fm' % avg_radius, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if pos > 0:
            cv2.putText(output, 'Distance From Center: %.2fm %s' % (np.absolute(pos), 'left'), (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(output, 'Distance From Center: %.2fm %s' % (np.absolute(pos), 'right'), (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if debug_flag:
            cv2.imshow("warped", warped)
            cv2.imshow("Binary", binary_warped*255)
            cv2.imshow("out", output)
            cv2.waitKey(100)
        # else:
        #     cv2.imshow("out", output)
        #     cv2.waitKey(1)

        return output

    def process_stream(self, path, save = False, debug_flag = False):
        import skvideo.io
        # open stream
        stream = skvideo.io.vread(path)
        cv2.waitKey(500)
        print("got stream")
        writer = skvideo.io.FFmpegWriter("result.mp4", outputdict={'-r': '10'})
        for frame in stream:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output = self.process_image(frame, debug_flag = debug_flag)
            if save:
                # write to video
                writer.writeFrame(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

        stream.release()
        cv2.destroyAllWindows()
        writer.close()

# create lane object
myLanes = detector(debug_duration=5)

# calibrate camera
myLanes.get_calibration(file_path='./camera_cal/calibration*.jpg', debug=False)

# process video
#myLanes.process_stream(path ='project_video.mp4', debug_flag=True,  save=False)

# get calibration images for report
img = cv2.imread('./camera_cal/calibration1.jpg')
undistorted = myLanes.undistort_image(image=img)
cv2.imwrite('./output_images/original.jpg', img)
cv2.imwrite('./output_images/undistorted.jpg', undistorted)

img = cv2.imread('./test_images/test4.jpg')
undistorted = myLanes.undistort_image(image=img)
cv2.imwrite('./output_images/original_lanes.jpg', img)
cv2.imwrite('./output_images/undistorted_lanes.jpg', undistorted)

# binary for report
binary = myLanes.color_thresh(undistorted, h_thresh=(10, 80), s_thresh=(75, 255), v_thresh=(75, 255))
cv2.imwrite('./output_images/binary_lanes.jpg', binary*255)

# warped for report
src = np.array([[585, 460], [203, 720], [1127, 720], [695, 460]], np.float32)
dst = np.array([[320, 0], [320, 720], [900, 720], [900, 0]], np.float32)
warped = myLanes.perspective_transform(undistorted, src, dst)
cv2.imwrite('./output_images/warped_lanes.jpg', warped)

warped_binary = myLanes.perspective_transform(binary, src, dst)
cv2.imwrite('./output_images/warped_lanes.jpg', warped_binary*255)

# final image for report
final = myLanes.process_image(image=img)
cv2.imwrite('./output_images/final_lanes.jpg', final)




