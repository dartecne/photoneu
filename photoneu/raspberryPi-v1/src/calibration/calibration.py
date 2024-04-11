from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
import serial

### globals ###
max_value = 255
low_V = 0
high_V = 90 # maximo valor bajo el cual se considera color negro
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'
port = '/dev/ttyACM0'
#baudrate = 115200
baudrate = 230400
filename = "data.dat"
state = 0 # 0 - stop, 1- moving, 
A = [1.1, 1.1] #coeficientes de calibración
B = [1.1, 1.1]

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

def endSystem():
    print(">>>exiting")
    ser.close()
    fd.close()
    exit()

def initSystem() :
    # configure the serial connections (the parameters differs on the device you are connecting to)
    global ser
    global fd
    ser = serial.Serial( port, baudrate )
    if ser.isOpen(): print(">>>Opened port: \"%s\"." % (port))
    else :
        print("Unable to open Serial Port: %s" % (port));
        print(">>>exiting")
        exit()
    fd = open(filename, "w")

def findHoughCircles(gray):
 circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=15,
                               minRadius=20, maxRadius=42) 
 if circles is not None:
        circles = np.uint16(np.around(circles))
        cv.putText(frame,"n_circles = " + str(circles.shape[0]),(20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(frame, center, radius, (255, 0, 255),1)    
            cv.putText(frame,str(radius),center, cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description
 return circles           

def findContours(frame, frame_threshold):
 x = -1
 y = -1
    # Find the contour in morphologyEx_img, and the contours are arranged according to the area from small to large.
 _tuple = cv.findContours( frame_threshold,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE ) 
 if len(_tuple) == 3:
     _, contours, hierarchy = _tuple
 else:
     contours, hierarchy = _tuple
 
 color_area_num = len(contours) # Count the number of contours

 if color_area_num > 0: 
     for i in contours:    # Traverse all contours
      approx = cv.approxPolyDP( 
        i, 0.01 * cv.arcLength(i, True), True)
#      print(approx)
      M = cv.moments(i) 
      if M['m00'] != 0.0: 
        x = int(M['m10']/M['m00']) 
        y = int(M['m01']/M['m00'])
      x,y,w,h = cv.boundingRect(i)      # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition 
      if w > 20 and h > 20 and abs(w-h) < 4 :       
         cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
         cv.putText(frame,str(x) + "," + str(y),(x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description
 return x, y

def sendCode( msg ) :
    print(">>> sendCode ")
    msg += "\0"
    ser.write( msg.encode(encoding= 'ascii') )
#    while ser.inWaiting() > 0:
#        print (".")
#        print ("<<<" + ser.readline().decode() )

def moveHead( x_head, y_head ):
    print( "moving to: " + str(x_head) + str(", ") + str(y_head) )
    msg = "X" + str(int(x_head)).zfill(5) + "Y" + str(int(y_head)).zfill(5) 
    sendCode(msg)
   
def getMotorPosition():   
    msg = "P\0"
    ser.write( msg.encode(encoding= 'ascii') )
    line = ser.readline().decode('utf-8').rstrip()
#    print("<<< reading... ")
#    print(line)
#    print(len(line.split(",")))
    timestamp, x_head, y_head = -1,-1,-1
    if len(line.split(",")) == 3:
        timestamp, x_head, y_head = line.split(",")
    if(timestamp == -1 ): 
        print("ERROR motor position")
        print("<<< reading... ")
        print(line)
    return int(timestamp), int(x_head), int(y_head)

def sendCalibrate() :
    msg = "C\0"
    ser.write( msg.encode(encoding= 'ascii') )    

def getSPerror() :
    msg = "E\0"
    ser.write( msg.encode(encoding= 'ascii') )    
    line = ser.readline().decode('utf-8').rstrip()
#    print("<<< reading... ")
#    print(line)
#    print(len(line.split(",")))
    timestamp, x_head_error, y_head_error = -1,-1,-1
    if len(line.split(",")) == 3:
        timestamp, x_head_error, y_head_error = line.split(",")
    if(timestamp == -1 ): 
        print("ERROR lectura SPerror")
        print("<<< reading... ")
        print(line)
    return int(timestamp), int(x_head_error), int(y_head_error)

def pixels2steps( point ) :  
   motor_point = [-1.0,-1.0] 
   motor_point[0] = int(A[0] * point[0] + B[0])
   motor_point[1] = int(A[1] * point[1] + B[1])

   return motor_point

######## main ##########

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)

initSystem()    
sendCalibrate() 
time.sleep(8)
#t, x_head, y_head = getMotorPosition()
x_head_0, y_head_0 = 0, 0
x_head_1, y_head_1 = 0, 0
x_cam_0, y_cam_0 = 0, 0
x_cam_1, y_cam_1 = 0, 0

######## loop  ##########
# enviar el cabezal al (0,0) y luego al centro
# buscar circulo rojo (cabezal) y guardar la posicion x0, y0
# mover el cabezal en x e y, y guardar la posición x1, y1
# obtener los coeficientes A, B

while True:
 ret, frame = cap.read()
 if frame is None:
     print( "No frame. Exit..." )
#     break
 frame = cv.blur(frame, (5,5))
 gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 rows = gray.shape[0]
 hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)              # Convert from BGR to HSV
 kernel = np.ones((6,6),np.uint8) 
 # filtro del rojo
 mask = cv.inRange( hsv,np.array([0, 60, 60]), np.array([14, 255, 255]) ) # red HSV: 0,4
 mask_2 = cv.inRange(hsv, (160,0,0), (180,255,255)) 
 mask = cv.bitwise_or(mask, mask_2)
 frame_threshold = cv.morphologyEx( mask, cv.MORPH_OPEN, kernel,iterations = 1 )              # Perform an open operation on the image 
 
# circles = findHoughCircles(frame_threshold)
# contours = findContours(frame_threshold)
 x_cam, y_cam =  findContours( frame, frame_threshold )
 
 cv.imshow( window_detection_name, frame_threshold )
 cv.imshow( window_capture_name, frame )
 line = ""

 if state == 0: # sin tomar la posicion 0
     t, x_head, y_head = getMotorPosition()
     line = str(t) + \
         "," + str(x_head) + "," + str(y_head) + \
         "," + str(x_cam) + "," + str(y_cam)
     print( "head, cam = " + line )
     t, x_head_error, y_head_error = getSPerror()
     print( "head_error = " + str(t) + "," + str(x_head_error) + "," + str(y_head_error) )

#     if (x_cam_0 != 0) & (y_cam_0 != 0) &\
#        (x_head_0 != 0) & (y_head_0 != 0):
     if (x_head_error == 0) & (y_head_error == 0) :
            print("Got point 0")
            x_cam_0 = x_cam
            y_cam_0 = y_cam
            x_head_0 = x_head
            y_head_0 = y_head
            print( "head_0, cam_0 = " + str(x_head_0) + "," + str(y_head_0) + "," + str(x_cam_0) + "," + str(y_cam_0) )
            moveHead( x_head + 10000, y_head + 10000 )
#            moveHead( 0, 0 )
            state = 1

 if state == 1:
    t, x_head_error, y_head_error = getSPerror()
    print( "head_error = " + str(t) + "," + str(x_head_error) + "," + str(y_head_error) )
    if (t != -1) & (x_head_error == 0) & (y_head_error == 0): # ya ha llegado
        print("Got SP")
        t, x, y = getMotorPosition()
        if t != -1:
            if( x_cam != 0 ) & ( y_cam != 0 ):
                x_cam_1 = x_cam
                y_cam_1 = y_cam
                x_head_1 = x
                y_head_1 = y
                print( "head_1, cam_1 = " + str(t) + "," + str(x_head_1) + "," + str(y_head_1) + "," + str(x_cam_1) + "," + str(y_cam_1) )
                state = 2 

 if state == 2:
#    head = A cam + B
    print("callibrating with:")
    print( "head_0, cam_0 = " + str(x_head_0) + "," + str(y_head_0) + "," + str(x_cam_0) + "," + str(y_cam_0) )
    print( "head_1, cam_1 = " + str(x_head_1) + "," + str(y_head_1) + "," + str(x_cam_1) + "," + str(y_cam_1) )
    #ERROR en el calculo de coeficintes
    A[0] = ( x_head_0 - x_head_1 ) / ( x_cam_0 - x_cam_1 )
    B[0] = x_head_0 - A[0] * x_cam_0
    A[1] = ( y_head_0 - y_head_1 ) / ( y_cam_0 - y_cam_1 )
    B[1] = y_head_0 - A[0] * y_cam_0
    print("calibración OK")
    print(A)
    print(B)
    state = 3 # supuestamente calibrado
    time.sleep(3) 
    #fd.write( line )

 if state == 3:
    x_cam_sp = 300
    y_cam_sp = 200
    head_point = pixels2steps([x_cam_sp, y_cam_sp])
    print("moving to pixels: " + str(x_cam_sp) + "," + str(y_cam_sp))
    print( head_point )
    moveHead( head_point[0], head_point[1] )
    state = 4

 key = cv.waitKey( 30 )

 if key == ord('q') or key == 27:
     break
    