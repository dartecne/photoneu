from __future__ import print_function
import cv2 as cv
from picamera.array import PiRGBArray
from picamera import PiCamera

import argparse
import numpy as np
import time
import serial

# main - programa principal que une visión y movimiento.
# El movimiento se realiza enviando mensajes serial con el formato XnnnnYmmmm
# TODO:
# - mejorar la deteccion del raton para evitar falsos positivos
# - evitar el ruido de los motores
# - filtrar el movimiento para hacerlo suave

def mapValue(vin, minin, maxin, minout, maxout ) :
    if(vin > maxin): vin = maxin
    if(vin < minin): vin = minin
    vin = (vin - minin)*(maxout - minout)/(maxin - minin)+ minout
    
    return vin
    
def initSystem() :
    # configure the serial connections (the parameters differs on the device you are connecting to)
    global ser
    ser = serial.Serial( port, baudrate )
    if ser.isOpen(): print(">>>Opened port: \"%s\"." % (port))
    else :
        print("Unable to open Serial Port: %s" % (port));
        print(">>>exiting")
        exit()
    
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
 
def sendCodeTest( msg ) :
    print(">>> writing... ")
    msg += "\0"
    try:
        ser.write( msg.encode(encoding= 'ascii') );
    except:
        print( "ERROR writing message")
    while ser.inWaiting() > 0:
        try: 
            str = ser.readLine().decode()
            print ("<<<" + str )
        except:
            print( "ERROR receiving message" )
            return 
def sendCode( msg ) :
    print(">>> writing... ")
    msg += "\0"
    ser.write( msg.encode(encoding= 'ascii') );
    while ser.inWaiting() > 0:
        print ("<<<" + ser.readline().decode() )
                
def controlLoop():
 global x_old, y_old
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
         if w >= 8 and h >= 8: # 
           cv.rectangle(frame,(x,y),(x+w,y+h),(120,20,200),2)  # Draw a rectangular frame
         if( (w * h > minMicePixelArea) & (w * h < maxMicePixelArea)):
           cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
           x,y,w,h = cv.boundingRect(i)
           text = "mice"# + str(contours.index(i))
           pointText = str(x) + ", " + str(y)
           cv.putText(frame, text,(x,y), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1)# Add character description
           cv.putText(frame, pointText ,(x,y+40), cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),1)# Add character description
           #print( "x = " + str(x) + "  y = " + str(y) + "w = " + str(w) + "  h = " + str(h) )
           x_head = 0
           x_head = mapValue( x, x_min, x_max, x_head_min, x_head_max  )
           y_head = 0
           y_head = mapValue( y, y_min, y_max, y_head_max, y_head_min  )
           #filtro para evitar sobresaltos y ruido
           if( (x_old != 0) 
                & (abs(x - x_old) < x_max_thres )
                & (abs(x - x_old) > x_min_thres)
                & (abs(y - y_old) < y_max_thres )
                & (abs(y - y_old) > y_min_thres)) :
                msg = "X" + str(int(x_head)).zfill(5) + "Y" + str(int(y_head)).zfill(5) 
                sendCode(msg)
                print(">>>" + msg)
                #time.sleep(0.04)
           x_old = x
           y_old = y
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
baudrate = 115200

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
x_min = 0  #limits in pixels
x_max = 469
y_min = 0
y_max = 349
x_head_min = 0
x_head_max = 19430
x_min_thres = 2 # umbral entre valores de pixeles para considerar que si hay  movimiento
x_max_thres = 120 
x_old = 0
y_head_min = 0
y_head_max = 24272
y_min_thres = 2
y_max_thres = 120
y_old = 0

minMicePixelArea = 140 * 70 # size in pixels of a mouse
maxMicePixelArea = 130 * 140 # size in pixels of a mouse

initSystem()
time.sleep(7)

while True:
    controlLoop()
    time.sleep(0.01)



