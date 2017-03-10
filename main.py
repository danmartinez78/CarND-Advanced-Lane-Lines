import lanes
import cv2
import matplotlib.pyplot as plt

# create lane object
myLanes = lanes.detector(debug_duration=2000)
# calibrate camera
myLanes.get_calibration(file_path='./camera_cal/calibration*.jpg', debug=False)
# process image
image = cv2.imread('./test_images/test2.jpg')
cv2.imshow("input_image", image)
cv2.waitKey(500)
cv2.destroyAllWindows()
myLanes.process_image(image, show_debug=False)
# process video
myLanes.process_stream(path ='challenge_video.mp4')
