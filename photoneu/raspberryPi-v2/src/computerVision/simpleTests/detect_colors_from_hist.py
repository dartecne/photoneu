from __future__ import print_function
import argparse
import cv2
import numpy as np
from array import array

from functions_old import *


parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()

color_dict = {'red':[0,4],'orange':[5,18],'yellow':[22,37],'green':[42,85],'blue':[92,110],'purple':[115,165],'red_2':[165,180]}  #Here is the range of H in the HSV color space represented by the color


cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)
#cv2.namedWindow(window_control_name)
#cv2.createTrackbar(percent_name, window_control_name , percent, 1000, on_percent_trackbar)

#while True:
imagen = cv2.imread('../img/mices_color_1.png')
assert imagen is not None, "file could not be read, check with os.path.exists()"
#imagen = cv2.medianBlur(imagen,3 ) #interesante para quitar ruido
img_mask = mask_colors(imagen)
_tuple = cv2.findContours(img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)      
    # compatible with opencv3.x and openc4.x
if len(_tuple) == 3:
   _, contours, hierarchy = _tuple
else:
   contours, hierarchy = _tuple
print( "Found " + str(len(contours)) + " contours" )
#cv2.drawContours( imagen, contours, -1, ( 0, 255, 0 ), 3 )
for cnt in contours:
   (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
   (x,y) = map(int, (x,y))
   cnt_color = array("i", imagen[y,x])
   cv2.putText(imagen,"mice",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,cnt_color,2)
     
#for i in contours:
   
cv2.imshow(window_capture_name, imagen)
cv2.imshow("Mask", img_mask)

#control_loop()

cv2.waitKey(0)
cv2.destroyAllWindows()
