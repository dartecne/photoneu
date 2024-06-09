import threading
import time
from datetime import datetime
import numpy as np
import cv2 as cv
import random
import logging

from camHandler import CamHandler
from motorHandler import MotorHandler

class Controller:
    def __init__(self):
        self.cam = CamHandler()
        self.motor = MotorHandler()
        self.motor_steps_thres = 10 # diferencia de pasos para que se envie comando de movimiento al motor
        self.p = np.zeros((2,4)) # points for linear regresion
        self.callibrated = False
        self.A = [-53.82, 54]
        self.B = [22008, 3665]
        self.r = [1.1, 1.1] #coeff de correlacion 
        self.point = np.array(6) # t_motor, point_motor, t_cam, point_cam 

        self.log_info = logging.getLogger("info")
        self.log_data = logging.getLogger("data")
        self.log_info.setLevel(logging.INFO)
        self.log_data.setLevel(logging.INFO)        
        formater= logging.Formatter('%(message)s')
        info_fh = logging.FileHandler('info.log')
        file_name = datetime.now().strftime("%Y_%m_%d_%I_%M_%S") + ".log"
        data_fh = logging.FileHandler(file_name)
        info_fh.setFormatter( formater )
        data_fh.setFormatter( formater )
        self.log_info.addHandler( info_fh )
        self.log_data.addHandler( data_fh )

    def set_target_color(self, color):
        self.cam.target_color = color

    def callibrate(self):
        self.set_target_color("red_2")
        msg="time_motor,motor_x,motor_y,time_cam,cam_x,cam_y"
        self.log_data.info( msg)
        p_head = [[19000, 6000], [19000, 18000],\
                  [2000, 18000], [2000, 6000],\
                    [9700, 12000]]

        for i in range( len(p_head) ):
            ts = time.clock_gettime_ns(0)
            self.motor.moveHead( p_head[i][0], p_head[i][1] )
            _, x, y = self.getPoint(i)
            time.sleep(0.5)
            i = i + 1
            if i > 2: 
                self.linearRegression()
            msg=""
            for i in range(5):
                msg += str(self.point[i]) + ","
            msg += str(self.point[5])
            self.log_data.info( msg )
        
        for i in range( 128 ):            
            ts = time.clock_gettime_ns(0)
            rand_x = random.randrange(2000, 19000, 1)
            rand_y = random.randrange(6000, 18000, 1)
            self.motor.moveHead( rand_x, rand_y )
            _, x, y = self.getPoint(i)
            self.linearRegression()
            msg=""
            for i in range(5): #self.point = tm, x, y, tc, self.cam.target.pos[0],self.cam.target.pos[1]
                msg += str(self.point[i]) + ","
            msg += str(self.point[5])
            self.log_data.info( msg )

        self.callibrated = True

    def getPoint(self, i):
        '''Get cam and motor points when header is stopped. And the times when each point has been detected'''
        print( "Getting point " + str(i))
        self.waitSP() # wait for motor to get the SP
        tm = time.clock_gettime_ns(0)
        print( "Got motor SP" )
        t, x, y = self.motor.getMotorPosition()
        #@TODO: este while ralentiza mucho la hebra
        while True: # wait cam to stabilize
            time.sleep(0.0001) # a ver si asi no ralentiza tanto
            if self.cam.target.is_moving == False:
                break 
        tc = time.clock_gettime_ns(0)
        print( "Got cam point" )
        if i < 2:
            self.p[i] = [self.cam.target.pos[0],self.cam.target.pos[1],x,y]
        else:
            self.p = np.append(self.p, [[self.cam.target.pos[0],self.cam.target.pos[1],x,y]], axis=0)
        self.point = tm, x, y, tc, self.cam.target.pos[0],self.cam.target.pos[1]
        print("reference points:")
        print(self.p)
        return t, x, y

    def moveMotorPixels( self, x_cam_sp, y_cam_sp ):
        if self.callibrated == False:
            print("Device not callibrated")
            return -1
        if( x_cam_sp < 0 ) & \
            ( y_cam_sp < 0 ) :
            return -2
        head_point = self.pixels2steps([x_cam_sp, y_cam_sp])
        if (head_point[0] - self.motor.x) < self.motor_steps_thres & \
            (head_point[1] - self.motor.y) < self.motor_steps_thres :
            return -3

        print("moving to pixels: " + str(x_cam_sp) + "," + str(y_cam_sp))
        print( head_point )
        self.motor.moveHead( head_point[0], head_point[1] )
        self.waitSP()
        print("Got SP")
        while True:
            if self.cam.target.is_moving == False :
                break
        error = str(self.cam.target.pos[0] - x_cam_sp)\
              +", " + str(self.cam.target.pos[1] - y_cam_sp)
        print("error in pixels: " + error)
        msg = str(head_point[0]) + "," + str(head_point[1]) + "," +\
                str(self.cam.target.pos[0]) + "," + str(self.cam.target.pos[1]) + "," + \
                error
        self.log_info.info( msg )

    def waitSP( self ):
        while True:
            t, x_head_error, y_head_error = self.motor.getSPerror()
#            self.motor.printValues(t, x_head_error, y_head_error)
            if (t!=-1) & \
                (x_head_error == 0) & \
                    (y_head_error == 0) :
                        self.motor.getMotorPosition() # ademas de get hace un update
                        break
    
    def linearRegression( self ):
#        self.p[i] = [x_cam,y_cam,x_motor,y_motor]
# motor_p = A * cam_p + B
        xm  = np.mean(self.p[:,0]) #x_cam[i]
        ym  = np.mean(self.p[:,2]) #x_motor[i]
        sx  = np.sum(self.p[:,0])
        sy  = np.sum(self.p[:,2])
        sxy = np.sum(self.p[:,0]*self.p[:,2])
        sx2 = np.sum(self.p[:,0]**2)
        sy2 = np.sum(self.p[:,2]**2)
        n = len(self.p)
        # coeficientes a0 y a1
        self.A[0] = (n*sxy-sx*sy)/(n*sx2-sx**2)
        self.B[0] = ym - self.A[0]*xm
        numerador = n*sxy - sx*sy
        raiz1 = np.sqrt(n*sx2-sx**2)
        raiz2 = np.sqrt(n*sy2-sy**2)
        self.r[0] = numerador/(raiz1*raiz2)

        xm  = np.mean(self.p[:,1]) #y_cam[i]
        ym  = np.mean(self.p[:,3]) #y_motor[i]
        sx  = np.sum(self.p[:,1])
        sy  = np.sum(self.p[:,3])
        sxy = np.sum(self.p[:,1]*self.p[:,3])
        sx2 = np.sum(self.p[:,1]**2)
        sy2 = np.sum(self.p[:,3]**2)
        # coeficientes a0 y a1
        self.A[1] = (n*sxy-sx*sy)/(n*sx2-sx**2)
        self.B[1] = ym - self.A[1]*xm
        numerador = n*sxy - sx*sy
        raiz1 = np.sqrt(n*sx2-sx**2)
        raiz2 = np.sqrt(n*sy2-sy**2)
        self.r[1] = numerador/(raiz1*raiz2)
        print(self.A)
        print(self.B)
        print(self.r)
#        if n > 6: self.callibrated = True

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
    
