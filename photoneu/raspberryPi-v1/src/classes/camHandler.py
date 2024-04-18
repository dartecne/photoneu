from __future__ import print_function
import numpy as np
import cv2 as cv




class CamHandler:
    def __init__(self):
        self.window_capture_name = 'Video Capture'
        self.window_detection_name = 'Object Detection'
        self.low_V_name = 'Low V'
        self.high_V_name = 'High V'
        self.max_value = 255
        self.low_V = 0
        self.high_V = 20 # maximo valor bajo el cual se considera color negro


        self.cap = cv.VideoCapture( 0 )
        cv.namedWindow( self.window_capture_name )
        cv.namedWindow( self.window_detection_name )
        cv.createTrackbar( self.low_V_name, self.window_detection_name , \
                          self.low_V, self.max_value, self.on_low_V_thresh_trackbar )
        cv.createTrackbar( self.high_V_name, self.window_detection_name , \
                          self.high_V, self.max_value, self.on_high_V_thresh_trackbar )


    def getImage( self ):
        ret, frame = self.cap.read()
        if frame is None:
            print( "No frame. Exit..." )
            return -1
        frame = cv.blur(frame, (5,5))
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#        rows = gray.shape[0]
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)     # Convert from BGR to HSV
        kernel = np.ones((6,6),np.uint8) 
        # filtro del rojo
        mask = cv.inRange( hsv,(160, 60, 60), (180, 255, 255) ) 
        mask_2 = cv.inRange( hsv, (0, 90, 50), (18, 255, 255) ) # red HSV: 0,18
        mask = cv.bitwise_or( mask, mask_2 )
#        mask = cv.inRange( mask, self.low_V, self.high_V )
        frame_threshold = cv.morphologyEx( mask, cv.MORPH_OPEN, kernel,iterations = 2 )              # Perform an open operation on the image 
        return frame, hsv, gray, frame_threshold
    
    def showImage(self, frame, frame_threshold):
        cv.imshow( self.window_detection_name, frame_threshold )
        cv.imshow( self.window_capture_name, frame )

    def findHoughCircles(self, frame, gray):
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 40,
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

    def findContours(self, frame, frame_threshold):
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
        return color_area_num, x, y

    def printValues( self, x, y ):
#        n, x, y = self.findContours()
        line = str( x ) + "," + str( y )
        print( "cam = " + line )

    def on_high_V_thresh_trackbar( self, val ):
        self.high_V = val
        self.high_V = max( self.high_V, self.low_V+1 )
        cv.setTrackbarPos( self.high_V_name, self.window_detection_name, self.high_V )

    def on_low_V_thresh_trackbar(self, val):
        self.low_V = val
        self.low_V = min( self.high_V-1, self.low_V )
        cv.setTrackbarPos( self.low_V_name, self.window_detection_name, self.low_V )