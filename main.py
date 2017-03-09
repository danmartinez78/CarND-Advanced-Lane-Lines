import lanes
import cv2

# create lane object
myLanes = lanes.detector()
# calibrate camera
myLanes.get_calibration(file_path='./camera_cal/calibration*.jpg', debug=False)
# process image
image = cv2.imread('./test_images/test1.jpg')
cv2.imshow('input image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
myLanes.process_image(image)
# process frames

# write video
