from __future__ import print_function
import threading
import numpy as np
import cv2 as cv



class Target:
    def __init__(self): 
        self.limits = (0,0)
        self.max_radius = 60
        self.color_id = ""
#        self.real_color = np.array([110, 100, 100])  #in HSV
        self.mean_color = 0
        self.pos = np.array([0, 0])
        self.vel = np.array([0, 0])
        self.pos_old = np.array([0, 0])
        self.is_moving = False
        self.is_moving_old = False
        self.is_tracked = False
        self.v_thres = 120 # number of pixels , its moving
        self.vel_mod = 0
        self.ratio = 0 # ratio of circle in pixels
#        self.tracker = cv.Tracker()
        self.update()

#@TODO:
    def track(self, x, y, r, c):
        if(x == -1) | (y == -1): 
            return
#        tracker = cv.Tracker()
#       is_initialized = tracker.init(frame, bounding_box)
#       is_tracking, bounding_box = tracker.update(frame)
        if (x > 0) & (x < self.limits[0]) & \
            (y > 0) & (y < self.limits[1]) & \
            (r > 0 ) & (r < self.max_radius):
            self.is_tracked = True
        else:
            self.is_tracked = False

    def setFeatures(self, x, y, r, c):
        pos = np.array([0,0])
        pos[0] = x
        pos[1] = y

        m = np.dot( pos - self.pos,\
                    pos - self.pos)
        if m > 6000:
            print("Target::New pos out of range: " + str(m))
            if self.is_tracked == False:
                self.track(x, y, r, c)
            else:
                return
        if self.is_tracked == True:
            self.pos[0] = x
            self.pos[1] = y
            self.ratio = r
            self.mean_color = c

    def update( self ):
#        print( "Target::head pos = " + str(self.pos) )
#        print( "Target::head pos_old = " + str(self.pos_old) )
#            print("CamHandler::v_head = " + str(self.header.vel))

        self.vel = self.pos - self.pos_old
        self.vel_mod = np.dot( self.vel, self.vel )

        if self.vel_mod > self.v_thres :
            self.is_moving = True
        else:
            self.is_moving = False
        
        if self.is_moving != self.is_moving_old:
            if self.is_moving == True: 
                print( ">>>>Target::target moving: " + \
                      str( self.vel[0] ) + ", " + str( self.vel[1] ) + ": " + str(self.vel_mod) )
            else: print( "<<<<Target::target stopped" )

        self.pos_old = np.copy( self.pos )
        self.is_moving_old = self.is_moving

class CamHandler:
    def __init__( self ):
        #@TODO: array de targets
        self.header = Target()
        self.header.color_id = "red"
        self.n_circles = 0
        self.window_capture_name = 'Video Capture'
        self.window_detection_name = 'Object Detection'
        self.low_V_name = 'Low V'
        self.high_V_name = 'High V'
        self.max_value = 255
        self.low_V = 0
        self.high_V = 20 # maximo valor bajo el cual se considera color negro
#@TODO: incluir color negro, para el raton
        self.color_h = {'red':[0,18],'orange':[5,18],'yellow':[22,37],'green':[42,85],'blue':[105,133],'purple':[115,165],'red_2':[160,180]}  #Here is the range of H in the HSV color space represented by the color
        self.color_s = {'red':[20,255], 'red_2':[20,255], 'blue':[122,255]}  
        self.color_v = {'red':[60,180],'red_2':[60,255],'blue':[60,255]}

        self.stop = False
        self.cap = cv.VideoCapture( 0 )

        cv.namedWindow( self.window_capture_name )
        cv.namedWindow( self.window_detection_name )
        cv.createTrackbar( self.low_V_name, self.window_detection_name , \
                          self.low_V, self.max_value, self.on_low_V_thresh_trackbar )
        cv.createTrackbar( self.high_V_name, self.window_detection_name , \
                          self.high_V, self.max_value, self.on_high_V_thresh_trackbar )
        self.control_thread = threading.Thread( target = self.controlLoop, args=(1,) )
        self.control_thread.start()

    def endSystem( self ):
        print( "CamHandler::exiting" )
        self.stop = True
        cv.destroyAllWindows()
        exit()

    def controlLoop( self, name ):
        while True:
            frame, hsv, gray = self.getImage()
            self.header.limits = frame.shape
            #@TODO: añadir deteccion de todos los colores
            frame_threshold = self.filterColor( hsv, self.header.color_id)
            #circles = cam.findHoughCircles(frame, gray)
            self.n_circles, x, y, r, c = self.findContours(frame, frame_threshold)
            self.header.setFeatures(x, y, r, c)
            
            cv.putText( frame,"n_circles = " + str(self.n_circles),(20,20), \
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,(100,100,150),2)

            cv.putText( frame,"v_head = " + str(self.header.vel_mod),(20,40), \
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,(100,100,150),2)

            cv.putText( frame,"head_color = " + str(self.header.mean_color),(20,60), \
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,(100,100,150),2)
            
            self.showImage( frame, frame_threshold)
            if self.n_circles < 1:
                print("CamHandler::no circles!")
                continue
            print("CamHandler::head pos = " + str(self.header.pos))

            self.header.update()
            print("CamHandler::head pos old = " + str(self.header.pos_old))
            print("CamHandler::v_head = " + str(self.header.vel))
            if self.stop == True:
                print("CamHandler::stopping control thread")
                break

    def getImage( self ):
        ret, frame = self.cap.read()
        if frame is None:
            print( "No frame. Exit..." )
            return -1
        frame = cv.blur(frame, (5,5))
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#        rows = gray.shape[0]
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)     # Convert from BGR to HSV
        return frame, hsv, gray
    
    def filterColor( self, hsv, color ):
        # filtro del rojo
        mask = cv.inRange( hsv,( self.color_h[color][0], \
                                self.color_s[color][0], \
                                self.color_v[color][0]), \
                          (self.color_h[color][1], 255, 255) ) 
        if color == 'red':
            mask_2 = cv.inRange( hsv, ( self.color_h['red_2'][0], \
                                        self.color_s['red_2'][0],\
                                        self.color_v['red_2'][0]), \
                                ( self.color_h['red_2'][1], 255, 255) ) # red HSV: 0,18
            mask = cv.bitwise_or( mask, mask_2 )
#        mask = cv.inRange( mask, self.low_V, self.high_V )
        kernel = np.ones((6,6),np.uint8) 
        frame_threshold = cv.morphologyEx( mask, cv.MORPH_OPEN, kernel, iterations = 2 )              # Perform an open operation on the image 
        return frame_threshold

    def showImage(self, frame, frame_threshold):
        cv.imshow( self.window_detection_name, frame_threshold )
        cv.imshow( self.window_capture_name, frame )
        key = cv.waitKey( 30 )
        if key == ord('q') or key == 27:
            self.endSystem()

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
        mean_color = 0
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
                    area = cv.contourArea(i)
                    x,y,w,h = cv.boundingRect(i)      # Decompose the c#ontour into the coordinates of the upper left corner and the width and height of the recognition 
 #                   if cv.contourArea(i) < 600 & cv.contourArea > 300:
                    (c_x,c_y), radius = cv.minEnclosingCircle(i)
                    center = (int(c_x), int(c_y))
                    radius = int(radius)
                    if w > 20 and h > 20 and abs(w-h) < 4 :    
                        mask = np.zeros(frame.shape, np.uint8)
                        cv.drawContours(mask,[i],0,255,-1)
                        pixelpoints = np.transpose(np.nonzero(mask))
#                        mean_color = cv.mean(frame, mask = mask)
                        mean_color = cv.mean( pixelpoints )
#                        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
                        cv.circle(frame,center,radius,(0,255,100),1)
                        cv.putText(frame,str(x) + "," + str(y),(x,y), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
                        cv.putText(frame,str(cv.contourArea(i)),(x,y + 60), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)

        return color_area_num, x, y, radius, mean_color


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