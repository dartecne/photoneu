from __future__ import print_function
import cv2 as cv
from picamera.array import PiRGBArray
from picamera import PiCamera

import argparse
import numpy as np
import time
import serial

# main - programa principal quie une visión y movimiento
# TODO:
# - mejorar la deteccion del raton para evitar falsos positivos
# - evitar el ruido de los motores
# - filtrar el movimiento para hacerlo suave
# 	- testar otras funciones G01 G02 G03
# 	- hacer el código en Arduino de movimiento de la máquina


def mapValue(vin, minin, maxin, minout, maxout ) :
    if(vin > maxin): vin = maxin
    if(vin < minin): vin = minin
    vin = (vin - minin)*(maxout - minout)/(maxin - minin)+ minout
    
    return vin
    
def initSystem() :
    sendCode( '$' )
    sendCode( '$X' )
    sendCode( 'G90' ) # coordenadas absolutas
    
def endSystem():
    print(">>>exiting")
    ser.close()
    exit()

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
 
def sendCode( msg ) :
    print(">>>writing... ")
    ser.write( msg.encode() );
    ser.write('\n'.encode())
    print(">>>serial says: ")
    while ser.inWaiting() > 0:
        print (ser.readline().decode())
                
def controlLoop():
 global x_head_old, y_head_old
 ret, frame = cap.read()
 if frame is None:
     return
# print( frame.shape)
 frame = frame[x_crop_min:frame.shape[0]-x_crop_max, y_crop_min:frame.shape[1]-y_crop_max]
 frame_GRAY = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 frame_GRAY = cv.blur(frame_GRAY, (3,3))
 
 frame_threshold = cv.inRange( frame_GRAY, low_V, high_V )
 frame_threshold = cv.morphologyEx( frame_threshold, cv.MORPH_OPEN, kernel_5,iterations=1 )
 _tuple = cv.findContours(frame_threshold,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)      
    # compatible with opencv3.x and openc4.x
 if len(_tuple) == 3:
        _, contours, hierarchy = _tuple
 else:
        contours, hierarchy = _tuple
    
 color_area_num = len(contours) # Count the number of contours

 if color_area_num > 0: 
     for i in contours:    # Traverse all contours
         # TODO: descartar lo que no sea raton poniendo un rango de w y h
         x,y,w,h = cv.boundingRect(i)      # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition object

            # Draw a rectangle on the image (picture, upper left corner coordinate, lower right corner coordinate, color, line width)
         if w >= 8 and h >= 8: # Because the picture is reduced to a quarter of the original size, if you want to draw a rectangle on the original picture to circle the target, you have to multiply x, y, w, h by 4.
           cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
           cv.putText(frame,"mice",(x,y), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)# Add character description

 x,y,w,h = cv.boundingRect(contours[0])
 print( "x = " + str(x) + "  y = " + str(y) )
 x_head = 0.0
 x_head = mapValue( x, x_min, x_max, x_head_min, x_head_max  )
 y_head = 0.0
 y_head = mapValue( y, y_min, y_max, y_head_min, y_head_max  )
 x_head = round( x_head, 2 )
 y_head = round( y_head, 2 )
 if( abs(x_head - x_head_old) > x_head_thres ) : x_head = x_head_old
 if( abs(y_head - y_head_old) > y_head_thres ) : y_head = y_head_old

 msg = 'G0 X'
 msg += str(x_head)
 msg += ' Y'
 msg += str(y_head)
 msg += ' Z'
 msg += str(y_head)
 
 sendCode(msg)
 print(msg)
 x_head_old = x_head
 y_head_old = y_head
 cv.imshow(window_capture_name, frame)
 cv.imshow(window_detection_name, frame_threshold)
 
 key = cv.waitKey(30)

 if key == ord('q') or key == 27:
     endSystem()
 
#### main ####
max_value = 255
low_V = 0
high_V = 20 # maximo valor bajo el cual se considera color negro
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'

port = '/dev/ttyACM0'
#port = '/dev/ttyACM1'
baudrate = 115200
# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial( port, baudrate )
ser.setDTR(0)
if ser.isOpen(): print(">>>Opened port: \"%s\"." % (port))
out = ''

ser.write("\r\n\r\n".encode());
time.sleep(2)
#ser.flushInput()
x_crop_min = 30
x_crop_max = 30
y_crop_min = 30
y_crop_max = 30
kernel_5 = np.ones((3,3),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
x_min = 48.0  #limits in pixels
x_max = 445.0
y_min = 0.0
y_max = 344.0
x_head_min = 0.0
x_head_max = -4.0
x_head_thres = 1
x_head_old = 0.0
y_head_min = 0.0
y_head_max = 6.0
y_head_thres = 1
y_head_old = 0.0

initSystem()

while True:
    controlLoop()
    time.sleep(0.05)
import cv2
