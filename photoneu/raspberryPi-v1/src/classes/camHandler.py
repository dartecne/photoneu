from __future__ import print_function
import threading
import numpy as np
import cv2 as cv
import math


class Tracker():
    """
    This class represents a tracker object that uses OpenCV and Kalman Filters.
    """
# https://pieriantraining.com/kalman-filter-opencv-python-example/

    def __init__(self, id, track_window):
        """
        Initializes the Tracker object.

        Args:
            id (int): Identifier for the tracker.
            hsv_frame (numpy.ndarray): HSV frame.
            track_window (tuple): Tuple containing the initial position of the tracked object (x, y, width, height).
        """

        self.id = id
        self.processNoise = 1e-1#3e-5
        self.measureNoise = 1e-1#1e-2

        self.kalman = cv.KalmanFilter(4, 2, 0)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        
        self.kalman.processNoiseCov = np.array(
            [[1., 0, 0, 0],
             [0, 1., 0, 0],
             [0, 0, 1., 0],
             [0, 0, 0, 1.]], np.float32) * self.processNoise

#        self.kalman.measurementNoiseCov = np.array(
#            [[1., 0], 
#             [0, 1.]], dtype=np.float32) * self.measureNoise
        

        x, y, w, h = track_window
        cx = x+w/2
        cy = y+h/2
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)

    def update( self, target):
        prediction = self.kalman.predict()
        if target.is_tracked == True:
            x, y = target.measured_pos[0], target.measured_pos[1]
            measurement = np.array([x, y], dtype= np.float32)
            self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        target.pos[0] = prediction[0]
        target.pos[1] = prediction[1]
#        target.vel = int(prediction[2]), int(prediction[3])

        return prediction


class Target:
    def __init__(self): 
        self.color_id = ""
#        self.real_color = np.array([110, 100, 100])  #in HSV
#        self.pos_limits = np.array([78, 430, 64, 300])
        self.pos_limits = np.array([78, 430, 50, 300])
        self.mean_color = 0
        self.measured_pos = np.array([0, 0]) # posicion dada por la camara
        self.pos = np.array([0, 0]) # posicion filtrada
        self.vel = np.array([0, 0]) # velocidad filtrada
        self.pos_old = np.array([0, 0])
        self.area = 0
        self.is_moving = False
        self.is_moving_old = False
        self.is_tracked = False
        self.vel_mod = 0.0
        self.v_thres = 4
        self.radius = 0 # ratio of circle in pixels
        self.boundingRect = None
        self.roi = None
        self.tracker = Tracker(0, (200,300,20,20))
        self.n_stopped_frames = 0


#        self.track_window = track_window
#        self.term_crit = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 1)
#        x, y, w, h = track_window
#        roi = hsv_frame[y:y+h, x:x+w]
#        roi_hist = cv.calcHist([roi], [0, 2], None, [15, 16],[0, 180, 0, 256])
#        self.roi_hist = cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

        
    def update( self ):
#        print( "Target::head pos = " + str(self.pos) )
#        print( "Target::head pos_old = " + str(self.pos_old) )
        self.pos[0] = self.measured_pos[0]
        self.pos[1] = self.measured_pos[1]
#        prediction = self.tracker.update( self )  
        self.pos[0] = max(self.pos[0], self.pos_limits[0])
        self.pos[0] = min(self.pos[0], self.pos_limits[1])
        self.pos[1] = max(self.pos[1], self.pos_limits[2])
        self.pos[1] = min(self.pos[1], self.pos_limits[3])

        self.vel[0] = self.pos[0] - self.pos_old[0]
        self.vel[1] = self.pos[1] - self.pos_old[1]
        self.vel_mod = self.vel[0] * self.vel[0] + \
            self.vel[1] * self.vel[1]
#        print("CamHandler::v_head = " + str(self.vel))
#        self.vel_mod = int(math.sqrt(self.vel_mod))

        if self.vel_mod > self.v_thres :
            self.n_stopped_frames = 0
        else:
            self.n_stopped_frames += 1
#        if self.is_moving != self.is_moving_old:
        if self.n_stopped_frames < 4:
            self.is_moving = True
#            print( ">>>>Target::target moving: " + \
#                      str( self.vel[0] ) + ", " + str( self.vel[1] ) + ": " + str(self.vel_mod) )
                
        elif self.n_stopped_frames >= 4: 
            self.is_moving = False
 #           print( "<<<<Target::target stopped" )
#
        self.pos_old = np.copy( self.pos )
        self.is_moving_old = self.is_moving

class CamHandler:
    def __init__( self ):
        self.target = Target()
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
        self.target_color = "red_2"
        self.cap = cv.VideoCapture( 0 )
        self.backSub = cv.createBackgroundSubtractorMOG2()
        self.fgMask = None

        cv.namedWindow( self.window_capture_name )
        cv.namedWindow( self.window_detection_name )
        cv.createTrackbar( self.low_V_name, self.window_detection_name , \
                          self.low_V, self.max_value, self.on_low_V_thresh_trackbar )
        cv.createTrackbar( self.high_V_name, self.window_detection_name , \
                          self.high_V, self.max_value, self.on_high_V_thresh_trackbar )
        self.stop_thread = False
        print( "CamHandler::ctor" )
        print( "CamHandler::beginning thread" )
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
            #@TODO: añadir deteccion de todos los colores
            frame_threshold = self.filterColor( hsv, self.target_color)
            fc = self.findContours(frame, frame_threshold)
            self.matchTargets(fc)
            self.target.update()

            x, y, radius = self.target.measured_pos[0], self.target.measured_pos[1],\
                        self.target.radius  

            cv.putText( frame,"v_head = " + str(self.target.vel_mod),(20,40), \
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,(100,255,0),1)
            color = (0,100,255)
            if(self.target.is_tracked): 
                color = (100,255,0)

            cv.putText( frame,"tracked: " + str(self.target.is_tracked),(150,40), \
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if self.target.is_tracked == True:
                cv.circle(frame,(x,y),radius,(0,255,100),1)
                cv.putText(frame,str(x) + "," + str(y),(x+20,y-20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
                cv.putText(frame,str(self.target.area),(x+20,y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
            
            (x, y) = (int(self.target.pos[0]), int(self.target.pos[1]))
            cv.circle(frame, (x, y), 5, (255, 255, 0), 2)
#            cv.putText(frame,str(x) + "," + str(y),(x,y+5), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
            self.showImage( frame, frame_threshold)

            if self.stop_thread == True:
                print("CamHandler::stopping control thread")
                break

    def getImage( self ):
        x_crop_min = 70
        x_crop_max = 40
        y_crop_min = 50
        y_crop_max = 100
        ret, frame = self.cap.read()
        frame = frame[x_crop_min:(frame.shape[0]-x_crop_max), \
                      y_crop_min:(frame.shape[1]-y_crop_max)]        
        if frame is None:
            print( "No frame. Exit..." )
            return -1
        frame = cv.blur(frame, (5,5))
        #denoising, no solo no funciona sino que genera mucho retardo
#        if len(self.frames) < 5:
#            self.frames.append(frame)
#        else:
#            _ = self.frames.pop(0)
#            self.frames.append(frame)
#            frame = cv.fastNlMeansDenoisingMulti(self.frames, 2, 5, None, 4, 7, 1)
        self.fgMask = self.backSub.apply(frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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
        frame_threshold = cv.morphologyEx( mask, cv.MORPH_OPEN, kernel, iterations = 2 )  
                    # Perform an open operation on the image 
        return frame_threshold

    def showImage(self, frame, frame_threshold):

        cv.imshow( self.window_detection_name, frame_threshold )
        cv.imshow( self.window_capture_name, frame )
#        cv.imshow( "Filtro fondo", self.fgMask )
        key = cv.waitKey( 30 )
        if key == ord('q') or key == 27:
            self.endSystem()

    def findContours(self, frame, frame_threshold):
        # Find the contour in morphologyEx_img, and the contours are arranged according to the area from small to large.
        contours, hier = cv.findContours( frame_threshold,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE ) 
        filter_contours = [] 
        num_contours = len(contours) # Count the number of contours

        if num_contours > 0: 
            for c in contours:    # Traverse all contours
                approx = cv.approxPolyDP( 
                    c, 0.01 * cv.arcLength(c, True), True)
    #      print(approx)
                area = cv.contourArea( c )
                if (area > 600) & (area < 900) :
                    filter_contours.append(c)
                cv.drawContours(frame, contours, -1, (200,255,0), 1)
        return filter_contours

    def matchTargets( self, contours):
        #@TODO: contour is target? yes, update target; no, create new target.
        if len(contours) == 0:
            self.target.is_tracked = False
        else:
            for c in contours:
                self.target.is_tracked = True
                M = cv.moments(c) 
                self.target.measured_pos[0] = int(M['m10']/M['m00']) 
                self.target.measured_pos[1] = int(M['m01']/M['m00'])
                self.target.area = cv.contourArea(c)
                self.target.boundingRect = cv.boundingRect(c)      # Decompose the c#ontour into the coordinates of the upper left corner and the width and height of the recognition 
                (c_x,c_y), self.target.radius = cv.minEnclosingCircle(c)
                self.target.radius= int(self.target.radius)


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