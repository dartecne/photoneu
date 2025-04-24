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
    """
    Respresenta un blob detectado como posible raton etiquetado con un color
    """
    def __init__(self): 
        self.color_id = ""
#        self.real_color = np.array([110, 100, 100])  #in HSV
#        self.pos_limits = np.array([78, 430, 64, 300])
        self.pos_limits = np.array([22, 350, 28, 450]) # xmin, xmax, ymin, ymax
        self.mean_color = (0,0,0) # Color del target en BGR 
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
        self.radius = 1 # ratio of circle in pixels
        self.boundingRect = None
        self.roi = None
        self.tracker = Tracker(0, (200,300,20,20))
        self.n_stopped_frames = 0
        self.pbm_total_time = 20 # tiempo total de tratamiento PBM que se debe aplicar
        self.pbm_time = 0 # tiempo parcial aplicado

#        self.track_window = track_window
#        self.term_crit = (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 10, 1)
#        x, y, w, h = track_window
#        roi = hsv_frame[y:y+h, x:x+w]
#        roi_hist = cv.calcHist([roi], [0, 2], None, [15, 16],[0, 180, 0, 256])
#        self.roi_hist = cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
        
    def update( self ):
        """
        Evalúa si el target se está moviendo o no. Actualiza los datos con el filtro de Kalman
        """
#        print( "Target::head pos = " + str(self.pos) )
#        print( "Target::head pos_old = " + str(self.pos_old) )
        self.pos[0] = self.measured_pos[0]
        self.pos[1] = self.measured_pos[1]
  #      prediction = self.tracker.update( self )  
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
    """
    Clase manejadora de la cámara.
    Instancia 4 targets (4 posibles ratones) de distintos colores
    """
    def __init__( self ):
        self.targets = [] #Target()
        self.window_capture_name = 'Video Capture'
        self.window_detection_name = 'Object Detection'
        self.low_V_name = 'Low V'
        self.high_V_name = 'High V'
        self.max_value = 255
        self.low_V = 0
        self.high_V = 20 # maximo valor bajo el cual se considera color negro
        self.img_path = r'C:\Users\inges\OneDrive - UDIT\src\photoneu\photoneu\raspberryPi-v1\src\classes\marker-less-frame.jpg'
#        self.color_ids = ['red', 'red_2', 'blue','green','yellow']
        self.color_ids = ['red', 'green', 'blue','dark_blue']
#        self.color_h = {'red':[0,18],'orange':[5,18],'yellow':[22,37],'green':[50,83],'blue':[105,133],'purple':[115,165],'red_2':[160,180]}  #Here is the range of H in the HSV color space represented by the color
#        self.color_s = {'red':[80,255], 'red_2':[20,255],'yellow':[22,255],'green':[42,255],'blue':[122,255]}  
#        self.color_v = {'red':[60,130],'red_2':[60,255],'yellow':[22,255],'green':[60,255],'blue':[60,255]}
#       Estos datos pueden calibrarse utilizando test_inRange.py
        self.color_h = {'red':[150,180],'green':[60,90],'blue':[87,111],'dark_blue':[114,165]}  #Here is the range of H in the HSV color space represented by the color
        self.color_s = {'red':[26,255],'green':[115,255],'blue':[130,255],'dark_blue':[140,255]}  
        self.color_v = {'red':[88,255],'green':[50,255],'blue':[68,255],'dark_blue':[93,255]}
        self.cap = cv.VideoCapture( 0 )
        self.backSub = cv.createBackgroundSubtractorMOG2()
        self.fgMask = None
        print( "CamHandler::ctor" )
        
    def init(self):
        """
        Crea una hebra de control en bucle cerrado 
        """
        t1 = Target()
        t1.is_tracked = False
        t1.color_id = "red"
        t1.mean_color = (0, 0, 255)
        t2 = Target()
        t2.color_id = "green"
        t2.mean_color = (0, 255, 0)
        t3 = Target()
        t3.color_id = "blue"
        t3.mean_color = (255,0 ,0)
        t4 = Target()
        t4.color_id = "dark_blue"
        t4.mean_color = (255,100 ,100)
        self.targets.append(t1)
        self.targets.append(t2)
#        self.targets.append(t3)
#        self.targets.append(t4)
#        cv.namedWindow( self.window_capture_name )
#        cv.namedWindow( self.window_detection_name )
#        cv.createTrackbar( self.low_V_name, self.window_detection_name , \
#                          self.low_V, self.max_value, self.on_low_V_thresh_trackbar )
#        cv.createTrackbar( self.high_V_name, self.window_detection_name , \
#                          self.high_V, self.max_value, self.on_high_V_thresh_trackbar )
        self.stop_thread = False
        self.control_thread = threading.Thread( target = self.controlLoop, args=(1,) )
        self.control_thread.start()
        print( "CamHandler::beginning thread" )
        
    def endSystem( self ):
        print( "CamHandler::exiting" )
        self.stop = True
        cv.destroyAllWindows()
        exit()

    def controlLoop( self, name ):
        while True:
            t0 = cv.getTickCount()
            i = 0
            frame, hsv, gray = self.getImage()
            t1 = cv.getTickCount()
            
            for target in self.targets:
                t1_1 = cv.getTickCount()
                frame_threshold = self.filterColor( hsv, target.color_id)
                t2 = cv.getTickCount()
                fc = self.findContours(frame, frame_threshold)
                t3 = cv.getTickCount()
                self.matchTarget(fc, target)
                target.update()
                t4 = cv.getTickCount()

                x, y, radius = target.measured_pos[0], target.measured_pos[1],\
                        target.radius  

                mice_label = target.color_id + "_mice v = "
                cv.putText( frame, mice_label + str(target.vel_mod),(40, 40 + 20 * i), \
                           cv.FONT_HERSHEY_SIMPLEX, 0.5,target.mean_color,1)
                color = (0,100,255)
                if(target.is_tracked): 
                    color = target.mean_color   #(100,255,0)
                cv.putText( frame,"tracked: " + str(target.is_tracked),(200,40 + 20*i), \
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
                if target.is_tracked == True:
                    cv.circle(frame,(x,y),radius,color,1)
                    _x = x - 30
#                    if( y <= 2000): _y = y + 20
                    cv.putText(frame,str(x) + "," + str(y),(_x,y-20), cv.FONT_HERSHEY_SIMPLEX, 0.5,color,1)
                    cv.putText(frame,str(target.area),(x-20,y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5,color,1)
            
                (x, y) = (int(target.pos[0]), int(target.pos[1]))
                cv.circle(frame, (x, y), 5, (255, 255, 0), 1) # circulo central del F. Kalman
#                cv.putText(frame,str(x) + "," + str(y),(x,y+5), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1)
                i += 1
                self.showImage( frame, frame_threshold)
                t5 = cv.getTickCount()
                t_read = (t1 - t0) / cv.getTickFrequency() # t-leer la imagen
                t_filter = (t2 - t1_1) / cv.getTickFrequency() # t - filtro color
                t_contour = (t3 - t2 ) / cv.getTickFrequency() # 
                t_target = (t4 - t3)/ cv.getTickFrequency() # target update - kalman
                t_show = (t5 - t4) / cv.getTickFrequency()
 #               print(target.color_id + ": " + str(t_read)+ \
 #                   ", " + str(t_filter) + \
 #                   ", " + str(t_contour) + \
 #                   ", " + str(t_target) + \
 #                   ", " + str(t_show))
            if self.stop_thread == True:
                print("CamHandler::stopping control thread")
                break

    def getImage( self ):
        x_crop_min = 0#70
        x_crop_max = 0#40
        y_crop_min = 140
        y_crop_max = 140
        ret, frame = self.cap.read()
        #frame = cv.imread(self.img_path) 
        if frame is None:
            print( "No frame. Exit..." )
            return -1
        frame = frame[x_crop_min:(frame.shape[0]-x_crop_max), \
                      y_crop_min:(frame.shape[1]-y_crop_max)]       
#        print(frame.shape) # (480, 360, 3)
        frame = cv.medianBlur(frame, 5)
        frame = cv.blur(frame, (5,5))
        #denoising, no solo no funciona sino que genera mucho retardo
#        if len(self.frames) < 5:
#            self.frames.append(frame)
#        else:
#            _ = self.frames.pop(0)
#            self.frames.append(frame)
#            frame = cv.fastNlMeansDenoisingMulti(self.frames, 2, 5, None, 4, 7, 1)
#        self.fgMask = self.backSub.apply(frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)     # Convert from BGR to HSV

        return frame, hsv, gray
    
    def filterColor( self, hsv, color ):
        mask = cv.inRange( hsv,( self.color_h[color][0], \
                                self.color_s[color][0], \
                                self.color_v[color][0]), (self.color_h[color][1], self.color_s[color][1], self.color_v[color][1]))
#        if color == 'red':
#            mask_2 = cv.inRange( hsv, ( self.color_h['red_2'][0], \
#                                        self.color_s['red_2'][0],\
#                                        self.color_v['red_2'][0]), \
#                                ( self.color_h['red_2'][1], 255, 255) ) # red HSV: 0,18
#            mask = cv.bitwise_or( mask, mask_2 )
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
#        kernel = np.ones((6,6),np.uint8) 
        frame_threshold = cv.morphologyEx( mask, cv.MORPH_OPEN, kernel, iterations = 1 )  
        cv.imshow(color, frame_threshold)
        return frame_threshold

    def showImage(self, frame, frame_threshold):
        cv.imshow( self.window_detection_name, frame_threshold )
        cv.imshow( self.window_capture_name, frame )
#        cv.imshow( "Filtro fondo", self.fgMask )
        key = cv.waitKey( 30 )
        if key == ord('q') or key == 27:
#            cv.imwrite("frame.jpg", frame)
#            cv.imwrite("frame_threshold.jpg", frame_threshold)
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
                area = cv.contourArea( c )
#                if (area > 60) & (area < 1400) :
                if (area > 300) & (area < 1400) :
                    filter_contours.append(c)
                else:
                    print("Contour area out of bonds:" + str(area))
#                cv.putText(frame, str(area), (20,100), cv.FONT_HERSHEY_SIMPLEX, 0.5,(200,255,0),1)
                cv.drawContours(frame, contours, -1, (200,255,0), 1)
        return filter_contours

    def matchTarget( self, contours, target):
        """
        Actualiza los valores de los targets según los blobls detectados de colores 
        """
        #@TODO: contour is target? yes, update target; no, create new target.
#        print(target.color_id)
        if len(contours) == 0:
            target.is_tracked = False
        else:
            if len(contours) > 1:
                print("matchTarget:: WARN: found more than 1 contour for same color!")
            c = contours[0]
            target.is_tracked = True
            M = cv.moments(c) 
            target.measured_pos[0] = int(M['m10']/M['m00']) 
            target.measured_pos[1] = int(M['m01']/M['m00'])
            target.area = cv.contourArea(c)
            target.boundingRect = cv.boundingRect(c)      # Decompose the c#ontour into the coordinates of the upper left corner and the width and height of the recognition 
            (c_x,c_y), target.radius = cv.minEnclosingCircle(c)
            target.radius= int(target.radius)

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