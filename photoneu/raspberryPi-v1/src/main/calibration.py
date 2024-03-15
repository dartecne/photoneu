from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time

max_value = 255
low_V = 0
high_V = 90 # maximo valor bajo el cual se considera color negro
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'

kernel_5 = np.ones((3,3),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.

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

#@TODO: buscar un crculo de un color dado:
# O bien filtrar por color y del resultado buscar circulos
# O bien buscar circulos y de los resultados buscar por color.

while True: 
 ret, frame = cap.read()
 if frame is None:
     print( "No frame. Exit...")
     break
 frame = cv.blur(frame, (8,8))
 gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 rows = gray.shape[0]
 # find circles
 circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=12, maxRadius=30) 
 if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, (255, 0, 255),1)    

 hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)              # Convert from BGR to HSV
 mask = cv.inRange(hsv,np.array([0, 60, 60]), np.array([4, 255, 255]) ) # red HSV: 0,4
 mask_2 = cv.inRange(hsv, (165,0,0), (180,255,255)) 
 mask = cv.bitwise_or(mask, mask_2)
 frame_threshold = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_5,iterations=1)              # Perform an open operation on the image 
 # Find the contour in morphologyEx_img, and the contours are arranged according to the area from small to large.
 _tuple = cv.findContours( frame_threshold,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE ) 
 # compatible with opencv3.x and openc4.x
 if len(_tuple) == 3:
     _, contours, hierarchy = _tuple
 else:
     contours, hierarchy = _tuple
 
 color_area_num = len(contours) # Count the number of contours

 if color_area_num > 0: 
     for i in contours:    # Traverse all contours
      approx = cv.approxPolyDP( 
        i, 0.01 * cv.arcLength(i, True), True)
      print(approx)
      M = cv.moments(i) 
      if M['m00'] != 0.0: 
        x = int(M['m10']/M['m00']) 
        y = int(M['m01']/M['m00'])
      x,y,w,h = cv.boundingRect(i)      # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition 
      if w > 12 and h > 12 and abs(w-h) < 2 :
         
         cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
         cv.putText(frame,str(x) + "," + str(y),(x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description

 cv.imshow(window_capture_name, frame)
 cv.imshow(window_detection_name, frame_threshold)
 
 key = cv.waitKey(30)

 if key == ord('q') or key == 27:
     break
    