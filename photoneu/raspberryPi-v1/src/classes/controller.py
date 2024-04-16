import time
import numpy as np
from camHandler import CamHandler
from motorHandler import MotorHandler

#@TODO: separar en hebras el reconocimiento por visión y la escritura en el motor

class Controller:
    def __init__(self):
        self.cam = CamHandler()
        self.motor = MotorHandler()
        self.p = np.empty((2,4))    
        self.callibrated = False
        self.A = [1.1, 1.1]
        self.B = [1.1, 1.1]

    def callibrate(self):
        # punto 0
        x_cam, y_cam = self.waitSP()
        t, x, y = self.motor.getMotorPosition()
        self.p = np.append(self.p, [[x_cam,y_cam,x,y]], axis=0)
        print(self.p)
        time.sleep(1)
        self.motor.moveHead( x + 2000, y + 2000 )

        # punto 1
        x_cam, y_cam = self.waitSP()
        print("Got SP")
        time.sleep(1)
        t, x, y = self.motor.getMotorPosition()
        self.p = np.append(self.p, [[x_cam,y_cam,x,y]], axis=0)
        print(self.p)

        self.calculateCoefficents()

    def testCallibration( self ):
        if self.callibrated == False:
            return -1
        x_cam_sp = 300
        y_cam_sp = 200
        head_point = self.pixels2steps([x_cam_sp, y_cam_sp])
        print("moving to pixels: " + str(x_cam_sp) + "," + str(y_cam_sp))
        print( head_point )
        self.motor.moveHead( head_point[0], head_point[1] )
        x_cam, y_cam = self.waitSP()
        print("Got SP")
        time.sleep(1)
        x_cam, y_cam = self.waitSP()
        print("error in pixels: " + str(x_cam - x_cam_sp)\
              +", " + str(y_cam - y_cam_sp))

    def waitSP( self ):
        t, x_head_error, y_head_error = self.motor.getSPerror()
        frame, hsv, gray, frame_threshold = self.cam.getImage()
        #circles = cam.findHoughCircles(frame, gray)
        n, x_cam, y_cam = self.cam.findContours(frame, frame_threshold)
        while True:
            frame, hsv, gray, frame_threshold = self.cam.getImage()
            #circles = cam.findHoughCircles(frame, gray)
            n, x_cam, y_cam = self.cam.findContours(frame, frame_threshold)
            self.cam.printValues(x_cam, y_cam)
            self.cam.showImage(frame, frame_threshold)
            t, x_head_error, y_head_error = self.motor.getSPerror()
            self.motor.printValues(t, x_head_error, y_head_error)
            self.cam.printValues(x_cam, y_cam)
            if (t!=-1) & \
                (x_head_error == 0) & \
                    (y_head_error == 0) & \
                        ( n < 1 ):
                break
        return x_cam, y_cam

    def calculateCoefficents( self ):
        x_cam_0 = self.p[0,0]
        y_cam_0 = self.p[0,1]
        x_head_0 = self.p[0,2]
        y_head_0 = self.p[0,3]
        x_cam_1 = self.p[1,0]
        y_cam_1 = self.p[1,1]
        x_head_1 = self.p[1,2]
        y_head_1 = self.p[1,3]
        self.A[0] = ( x_head_0 - x_head_1 ) / ( x_cam_0 - x_cam_1 )
        self.B[0] = x_head_0 - self.A[0] * x_cam_0
        self.A[1] = ( y_head_0 - y_head_1 ) / ( y_cam_0 - y_cam_1 )
        self.B[1] = y_head_0 - self.A[1] * y_cam_0           
        self.callibrated = True
        print( "calibración OK" )
        print( self.A )
        print( self.B )
    
    def pixels2steps( self, point ):
        motor_point = [-1.0,-1.0] 
        if self.callibrated:
            motor_point[0] = int(self.A[0] * point[0] + self.B[0])
            motor_point[1] = int(self.A[1] * point[1] + self.B[1])        
        return motor_point
    
