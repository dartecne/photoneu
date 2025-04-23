from __future__ import print_function
import cv2 as cv
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import argparse
import numpy as np
import time


max_value = 255
low_V = 0
high_V = 80 # maximo valor bajo el cual se considera color negro
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'

kernel_5 = np.ones((5,5),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.
kernel_3 = np.ones((3,3),np.uint8) #Define a 3×3 convolution kernel with element values of all 1.

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

###############
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
x_crop_min = 30
x_crop_max = 30
y_crop_min = 30
y_crop_max = 30
img_path = r'/home/ratoncillos/OneDrive/src/photoneu/dataset/deeplabcut/labeled-data-ordered/img0253.png'
while True: 
 e1 = cv.getTickCount()
# ret, frame = cap.read()
 frame = cv.imread(img_path)
 if frame is None:
     break
# print( frame.shape)
# frame = frame[x_crop_min:frame.shape[0]-x_crop_max, y_crop_min:frame.shape[1]-y_crop_max]
 frame = cv.resize(frame, (320, 240)) # frame.shape/4
# frame = cv.resize(frame, (160, 120)) # frame.shape/4
 frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 frame_norm = cv.normalize(frame_gray, None, alpha = 0, beta = 255, norm_type=cv.NORM_MINMAX)
 frame_blur = cv.medianBlur(frame_norm, 5)
# frame_blur = cv.medianBlur(frame_blur, 5)
# frame_blur = cv.medianBlur(frame_blur, 5)
# frame_threshold = cv.inRange( frame_GRAY, low_V, high_V )
# frame_threshold = cv.bilateralFilter(frame_threshold, 5, 350, 350)
 ret, frame_threshold = cv.threshold( frame_blur, high_V, 255, cv.THRESH_BINARY_INV )
# frame_edge = cv.Canny(frame_threshold, 10, 400)
 dist_transform = cv.distanceTransform(frame_threshold,cv.DIST_L2,5)
# frame_threshold = cv.blur(frame_threshold, (3,3))
# frame_threshold = cv.adaptiveThreshold( frame_threshold, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#        cv.THRESH_BINARY,63,0 )
 frame_erode = cv.erode(frame_threshold, kernel_3, iterations = 2)
# frame_threshold = cv.morphologyEx( frame_threshold, cv.MORPH_CLOSE, kernel_3,iterations= 1 )
 frame_close = cv.morphologyEx( frame_erode, cv.MORPH_CLOSE, kernel_3,iterations= 3 )
 frame_open_1 = cv.morphologyEx( frame_close, cv.MORPH_OPEN, kernel_5,iterations= 2 )
# frame_open_2 = cv.morphologyEx( frame_open_1, cv.MORPH_OPEN, kernel_3,iterations= 5 )
 _tuple = cv.findContours(frame_open_1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)      
    # compatible with opencv3.x and openc4.x
 if len(_tuple) == 3:
        _, contours, hierarchy = _tuple
 else:
        contours, hierarchy = _tuple
    
 color_area_num = len(contours) # Count the number of contours
 if color_area_num > 0: 
    for i in contours:    # Traverse all contours
         x,y,w,h = cv.boundingRect(i)      # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition object
#         print( "x = " + str(x) + "  y = " + str(y) )
         area = cv.contourArea(i)
         if (area > 500) & (area < 1200):
            rect = cv.minAreaRect(i) # center, size, angle
            cx, cy = rect[0]
#            width, height = rect.size
#            print( "w = " + str(width) + "  y = " + str(height) )
            box = cv.boxPoints(rect)
            box = np.intp(box)
#         if w >= 16 and w < 300 and h >= 16 and h < 300: # Because the picture is reduced to a quarter of the original size, if you want to draw a rectangle on the original picture to circle the target, you have to multiply x, y, w, h by 4.
#            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
            cv.circle(frame, (int(cx),int(cy)),3,(255,255,25))
            cv.putText(frame,"mice:"+str(int(area)),(x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description
            cv.drawContours(frame,[box],0,(255,0,25),2)
#    print( "x = " + str(x) + "  y = " + str(y) )
 e2 = cv.getTickCount()
 fps = cv.getTickFrequency() / (e2 - e1) 
 cv.putText(frame,"tau = " + f"{fps:.0f}" + "FPS",(10,20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,25,0),1)# Add character description
 print(int(fps))
 cv.imshow(window_capture_name, frame)
 cv.imshow("Dist Transform", dist_transform)
 cv.imshow("Frame Normalized", frame_norm)
 cv.imshow("Frame Erode (3x3)", frame_erode)
 cv.imshow("Frame Threshold: 80", frame_threshold)
# cv.imshow("frame blur", frame_blur)
 cv.imshow("Frame Opening (5x5) x 2", frame_open_1)
# cv.imshow("frame open 2", frame_open_2)
 cv.imshow("Frame Closing (3x3) x3", frame_close)

 key = cv.waitKey(30)

 if key == ord('q') or key == 27:
     break
    
