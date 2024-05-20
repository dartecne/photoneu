import threading
import time
import numpy as np
import cv2 as cv
import random

from camHandler import CamHandler
from motorHandler import MotorHandler

class Controller:
    def __init__(self):
        self.cam = CamHandler()
        self.motor = MotorHandler()
        self.p = np.zeros((2,4)) 
#        self.p = [] # number of points (in pixels) to correlate    
        self.callibrated = False
        self.A = [-53.82, 54]
        self.B = [22008, 3665]
        self.r = [1.1, 1.1] #coeff de correlacion 
#        self.motor_thread = threading.Thread( target = self.motor_thread, args=(1,) )
#       self.motor_thread.start()

    def callibrate(self):
        p_head = [[19000, 6000], [19000, 18000],\
                  [2000, 18000], [2000, 6000],\
                    [9700, 12000]]
        for i in range( len(p_head) ):
#            dx = random.randrange(-2000,2000,1)
#            dy = random.randrange(-2000,2000,1)
            self.motor.moveHead( p_head[i][0], p_head[i][1] )
            t, x, y = self.getPoint(i)
            time.sleep(2)
            i = i + 1
            if i > 2: 
                self.linearRegression()
        self.callibrated = True

    def getPoint(self, i):
        '''Get cam and motor points when header is stopped'''
        print( "Getting point " + str(i))
        self.waitSP()
        print( "Got SP" )
        t, x, y = self.motor.getMotorPosition()
        while self.cam.target.is_moving == True :
            time.sleep(0.02) #compensacion del retardo de la camara
        print( "Got cam point" )
        if i < 2:
            self.p[i] = [self.cam.target.pos[0],self.cam.target.pos[1],x,y]
        else:
            self.p = np.append(self.p, [[self.cam.target.pos[0],self.cam.target.pos[1],x,y]], axis=0)
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
        print("moving to pixels: " + str(x_cam_sp) + "," + str(y_cam_sp))
        print( head_point )
        self.motor.moveHead( head_point[0], head_point[1] )
        self.waitSP()
        print("Got SP")
        while self.cam.target.is_moving == True :
            time.sleep(0.02) #compensacion del retardo de la camara
#        time.sleep(1)
        print("error in pixels: " + str(self.cam.target.pos[0] - x_cam_sp)\
              +", " + str(self.cam.target.pos[1] - y_cam_sp))

    def waitSP( self ):
        while True:
            t, x_head_error, y_head_error = self.motor.getSPerror()
#            self.motor.printValues(t, x_head_error, y_head_error)
            if (t!=-1) & \
                (x_head_error == 0) & \
                    (y_head_error == 0) :
            #                time.sleep(2) # esperamos a que la imagen se estabilice
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
    
