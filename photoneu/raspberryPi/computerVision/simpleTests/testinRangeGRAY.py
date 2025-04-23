from __future__ import print_function
import cv2 as cv
import argparse
max_value = 255
low_V = 0
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'
def on_low_V_thresh_trackbar(val):
 global low_V
 global high_V
 low_V = val
 low_V = min(high_V-1, low_V)
 cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
 global low_V
 global high_V
 high_V = val
 high_V = max(high_V, low_V+1)
 cv.setTrackbarPos(high_V_name, window_detection_name, high_V)
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
while True:
 
 ret, frame = cap.read()
 if frame is None:
     break
 frame_GRAY = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 frame_GRAY = cv.blur(frame_GRAY, (6,6))
 frame_threshold = cv.inRange(frame_GRAY, low_V, high_V)
 
 
 cv.imshow(window_capture_name, frame)
 cv.imshow(window_detection_name, frame_threshold)
 
 key = cv.waitKey(30)
 if key == ord('q') or key == 27:
     break
    