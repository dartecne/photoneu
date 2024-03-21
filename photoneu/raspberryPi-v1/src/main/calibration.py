from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import serial
import time

max_value = 255
low_V = 0
high_V = 90 # maximo valor bajo el cual se considera color negro
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'
lower_red = np.array([100, 100, 100])
upper_red = np.array([180, 255, 255])

kernel_5 = np.ones((3,3),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.
param1 = 100
param2 = 10 # [5,14]

port = '/dev/ttyACM0'
baudrate = 115200

def on_param1(val):
    global param1
    param1 = val
    cv.setTrackbarPos("param1", window_detection_name, param1)

def on_param2(val):
    global param2
    param2 = val
    cv.setTrackbarPos("param2", window_detection_name, param2)

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
 
# Función para detectar todos los círculos
def detect_circles(blurred):
    global param1, param2
    rows = blurred.shape[0]
    circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, dp=1, minDist=25,
                               param1=param1, param2=param2, minRadius=10, maxRadius=30)
    # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
    #                            param1=100, param2=30,
    #                            minRadius=12, maxRadius=30) 
    return circles

# Función para filtrar círculos rojos
def filter_red_circles(frame, circles):
    red_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if r > 0:
            # Extraer el área del círculo
                roi = frame[y - r:y + r, x - r:x + r]
                print(roi.shape) # 58, 58, 3
                if roi.shape[0] * roi.shape[1] > 0 :
                    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
                    mean_hsv = np.mean(hsv_roi, axis=(0, 1))
                    std_hsv = np.std(hsv_roi, axis=(0, 1))
                    if ((mean_hsv[0] >= 0 and mean_hsv[0] <= 10) or 
                        (mean_hsv[0] >= 130 and mean_hsv[0] <= 180)):
                        if(std_hsv[0] > 50):
                            red_circles.append((x, y, r))
    return red_circles

def circles_then_color(frame, edges ):
        ''' Primero detecta círculos y luego selecciona circulos rojos
        '''
        circles = detect_circles(edges)
        if circles is not None:
            print(len(circles))
            circles_int = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles_int:
                cv.circle(frame, (x,y), r, (200, 200, 200), 1)        
            red_circles = filter_red_circles(frame, circles)
            if red_circles:
                for (x, y, r) in red_circles:
                    cv.circle(frame, (x, y), r, (0, 0, 255), 2)
#        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
                    cv.putText(frame,str(x) + "," + str(y),(x,y-20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description
                    cv.putText(frame,str(r),(x+20,y+20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description

def detect_red_circles(frame, mask):
    # Encontrar contornos de los círculos
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Inicializar lista de círculos
    circles = []

    # Iterar sobre los contornos encontrados
    for contour in contours:
        # Encontrar el círculo aproximado
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)

        # Si el contorno es un círculo
        if len(approx) > 8:
            # Encontrar el centro y radio del círculo
            (x, y), radius = cv.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Añadir el círculo a la lista
            circles.append((center, radius))

    return circles
def sendCode( msg ) :
    print(">>> writing... ")
    msg += "\0"
    ser.write( msg.encode(encoding= 'ascii') );
    while ser.inWaiting() > 0:
        print ("<<<" + ser.readline().decode() )

def loop():    
    ret, frame = cap.read()
    if frame is None:
        print( "No frame. Exit...")
        exit()
#    print(frame.shape): 480x640x3
            # Convertir la imagen a espacio de color HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(hsv, lower_red, upper_red)
    blurred = cv.GaussianBlur(mask, (5, 5), 0)
#    blurred = cv.blur(gray, (3,3))
#    edges = cv.Canny( gray, 100,50)
    if ret:
#        circles = detect_red_circles(frame, blurred)
        circles = detect_circles(blurred)
        if circles is not None:
            print(len(circles))
            circles_int = np.round(circles[0, :]).astype("int")
#            for (x, y, r) in circles_int:
        # Dibujar los círculos encontrados en el frame
            for (x, y, r) in circles_int:
#            for center, radius in circles:
                cv.circle(frame, (x,y), r, (0, 0, 255), 2)
                cv.putText(frame,str(x) + "," + str(y),(x,y-20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description
                cv.putText(frame,str(r),(x+20,y+20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)# Add character description

        cv.imshow(window_capture_name, frame)
        cv.imshow(window_detection_name, blurred)

#### main ####
parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar("param1", window_detection_name , param1,600, on_param1)
cv.createTrackbar("param2", window_detection_name , param2, 60, on_param2)
#cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

initSystem()
while True:
    loop()
    if cv.waitKey(1) & 0xFF == ord('q'):
        exit()

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv.destroyAllWindows()

 
    