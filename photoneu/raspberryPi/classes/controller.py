import threading
import time
import random
import math
from datetime import datetime
import numpy as np
import cv2 as cv
import random
import logging
import sklearn as skl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle 

from camHandler import CamHandler
from motorHandler import MotorHandler

class Controller:
    """
    Controlador principal. Instancia un manejador de la cámara y otro de los motores.
    """

    def __init__(self):
        self.cam = CamHandler()
        self.cam.init() # inicia la hebra de deteccion de targets por color
        self.motor = MotorHandler()
        self.motor_steps_thres = 10 # diferencia de pasos para que se envie comando de movimiento al motor
        self.p = np.zeros((2,4)) # points for linear regresion
        self.callibrated = False
        self.A = [-24, 56] #[-53.82, 54]
        self.B = [12511, -1107] #[22008, 3665]
        self.r = [1.1, 1.1] #coeff de correlacion 
        self.point = np.array(6) # t_motor, point_motor, t_cam, point_cam 
        self.x_model = pickle.load(open("../logs/linear_x.sav", 'rb')) 
        self.y_model = pickle.load(open("../logs/linear_y.sav", 'rb')) 

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

    def applyTreatment( self, id ):
        msg="time_stamp,target_id,motor_x,motor_y,cam_x,cam_y,duration"
        self.log_data.info( msg) # escribe cabecera
        t = self.cam.targets[id]
        while t.pbm_time < t.pbm_total_time:
            if (t.is_moving == False) & (t.is_tracked):
                x = t.pos[0]
                y = t.pos[1]
                head_point = self.moveMotorPixels( x, y )
                time.sleep(1)
                t0 = time.time()
                ts = 0
                self.pbmON()
                msg = str(t0) + "," + str(id) + "," + str(head_point[0]) + "," + str(head_point[1]) + "," + str(x) + "," + str(y) + ","        
                while t.is_moving == False:
                    t1 = time.time()
                    ts += t1 - t0
                    t0 = time.time()
                    time.sleep(0.01)
                t.pbm_time += ts
                msg += str(t.pbm_time)   
                self.pbmOFF()
        print("END of PBM for target " + str(id))

    def pbmON(self):
        print("PBM - ON")

    def pbmOFF(self):
        print("PBM - OFF")

    def set_target_color(self, color):
        self.cam.targets[0].color_id = color
        
    def latencyTest(self):
        n_points = 20 #
        r = 2000 # ratio distance in steps
        self.set_target_color("red")
        msg="time_motor,motor_x,motor_y,time_sp,time_cam,cam_x,cam_y"
        self.log_data.info( msg) # escribe cabecera
        x_min, x_max = 1500, 16000
        y_min, y_max = 1500, 23000
        for n in range(n_points):
            values_ok = False
            while values_ok == False:
                a = 2 * math.pi * random.random()
                x = r * math.cos(a)
                y = r * math.sin(a)
                if (x < x_max) & (x > x_min) & (y < y_max) & (y > y_min):
                    values_ok = True
            t0 = time.clock_gettime_ns(0)
            msg = str(t0) + "," + str(x) + "," + str(y) + ","            
            self.motor.moveHead( x, y )
            self.waitSP() # wait for motor to get the SP
            tm = time.clock_gettime_ns(0)
            msg += str(tm)
            print( "Got motor SP" )
            while True: # Espera a que la imagen se mueva
                if self.cam.targets[0].is_moving == True:
                    break 
            while True: # wait cam to stabilize
                time.sleep(0.0001) # a ver si asi no ralentiza tanto
                if self.cam.targets[0].is_moving == False:
                    break 
            tc = time.clock_gettime_ns(0)
            msg += str(tc) + ","
            msg += str(self.cam.targets[0].pos[0]) + "," + str(self.cam.targets[0].pos[1])
            print( "Got cam point" )
            self.log_data.info( msg )

            
    def callibrate(self):
        self.set_target_color("red")
        msg = "t_total, t_motor, t_cam_1, t_cam2,"
        msg+=" time_motor,motor_x,motor_y,time_cam,cam_x,cam_y"
        self.log_data.info( msg)
        x_min, x_max = 1500, 16000# 2000, 19000
        y_min, y_max = 1500, 23000#7000, 18000
        step = 2000

#        p_head = [[19000, 6000], [19000, 18000],[2000, 18000], [2000, 6000],[9700, 12000]]
#        x = np.linspace(x_min,x_max, n, dtype= int)
#        y = np.linspace(y_min,y_max, n, dtype = int)
#        p_head = np.concatenate([x,y]).reshape(2,n)
        p_head = np.mgrid[x_min:x_max:step, y_min:y_max:step].reshape(2,-1).T
        indexes = random.sample(range(len(p_head)), len(p_head))

#        for i in range(len(p_head)) :
        for i in indexes :
            t0 = cv.getTickCount()
            ts = time.clock_gettime_ns(0)
            self.motor.moveHead( p_head[i][0], p_head[i][1] )
            tmotor, tcam1, tcam2, t, x, y = self.getPoint(i)
#            time.sleep(2)
            t1 = cv.getTickCount()
            i = i + 1
            if i > 2: 
                self.linearRegression()
            msg=""
            t_total = (t1 - t0)/(cv.getTickFrequency())
            msg = str(t_total) + ","
            msg+= str(tmotor) + "," + str(tcam1)+","+str(tcam2)+","
            for i in range(5):
                msg += str(self.point[i]) + ","
            msg += str(self.point[5])
            self.log_data.info( msg )
        
        self.callibrated = True

    def getPoint(self, i):
        '''Get cam and motor points when header is stopped. And the times when each point has been detected'''
        print( "Getting point " + str(i))
        t0 = cv.getTickCount()
        self.waitSP() # wait for motor to get the SP
        t1 = cv.getTickCount()
        tm = time.clock_gettime_ns(0)
        print( "Got motor SP" )
        t, x, y = self.motor.getMotorPosition()
        #Ocurre que aunque el motor haya llegado puede que la imagen todavía esté quieta (no haya arrancado)
        while True: # Espera a que la imagen se mueva
            if self.cam.targets[0].is_moving == True:
                break 
        #@TODO: este while ralentiza mucho la hebra
        while True: # wait cam to stabilize
            time.sleep(0.0001) # a ver si asi no ralentiza tanto
#            print("Controller cam_pos: " + str(self.cam.target.pos[0])\
#                  + ", " + str(self.cam.target.pos[1]))
            if self.cam.targets[0].is_moving == False:
                break 
        t2 = cv.getTickCount()
        tc = time.clock_gettime_ns(0)
        print( "Got cam point" )
        if i < 2:
            self.p[i] = [self.cam.targets[0].pos[0],self.cam.targets[0].pos[1],x,y]
        else:
            self.p = np.append(self.p, [[self.cam.targets[0].pos[0],self.cam.targets[0].pos[1],x,y]], axis=0)
        self.point = tm, x, y, tc, self.cam.targets[0].pos[0],self.cam.targets[0].pos[1]
        print("reference points:")
        print(self.p)
        t3 = cv.getTickCount()
        tmotor = (t1 - t0)/cv.getTickFrequency()
        tcam1 = (t2 - t1)/cv.getTickFrequency()
        tcam2  = (t3 - t2)/cv.getTickFrequency()
        return tmotor, tcam1, tcam2, t, x, y

    def moveMotorPixels( self, x_cam_sp, y_cam_sp ):
        """
        Función principal de movimiento del cabezal 
        """
        if self.callibrated == False:
            print("Device not callibrated")
            return -1
        if( x_cam_sp < 0 ) & \
            ( y_cam_sp < 0 ) :
            return -2
        head_point = self.pixels2steps([x_cam_sp, y_cam_sp])
        if abs(head_point[0] - self.motor.x) < self.motor_steps_thres & \
            abs(head_point[1] - self.motor.y) < self.motor_steps_thres :
            return -3

        print("moving to pixels: " + str(x_cam_sp) + "," + str(y_cam_sp))
        print( head_point )
        self.motor.moveHead( head_point[0], head_point[1] )
        self.waitSP()
        print("Got SP")
        return head_point

##Seguimiento del cabezal por CV
#        while True:
#            if self.cam.targets[0].is_moving == False :
#                break
#        error = str(self.cam.targets[0].pos[0] - x_cam_sp)\
#              +", " + str(self.cam.targets[0].pos[1] - y_cam_sp)
#        print("Vision error in pixels: " + error)
#        msg = str(head_point[0]) + "," + str(head_point[1]) + "," +\
#                str(self.cam.targets[0].pos[0]) + "," + str(self.cam.targets[0].pos[1]) + "," + \
#                error
#        self.log_info.info( msg )
#
    def waitSP( self ):
        while True:
            t, x_head_error, y_head_error = self.motor.getSPerror()
#            self.motor.printValues(t, x_head_error, y_head_error)
            if (t!=-1) & \
                (x_head_error == 0) & \
                    (y_head_error == 0) :
                        self.motor.getMotorPosition() # ademas de get hace un update
                        break
    def map_value(self, x, in_min, in_max, out_min, out_max):
        y = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        return y

    def pixels2steps( self, point ):
        """
        Recibe un punto en pixeles y devuelve los steps para llegar a ese punto
        """
        # según logs/dataAnalysis.ipynb
        motor_x_min, motor_x_max = 1500, 15500
        motor_y_min, motor_y_max = 1500, 21500

        cam_x_min, cam_x_max = 68, 337
        cam_y_min, cam_y_max = 36, 414
        print(point)
        point[0] = self.map_value(point[0], cam_x_min, cam_x_max, 1.0, 0.0 )
        point[1] = self.map_value(point[1], cam_y_min, cam_y_max, 0.0, 1.0 )
        print(point)
        motor_point = [-1.0,-1.0] 
        if self.callibrated:
            x_out = self.x_model.predict( [[point[0], point[1]]])
            y_out = self.y_model.predict( [[point[0], point[1]]])    
        x_out[[0]] = self.map_value(x_out[[0]], 0.0, 1.0, motor_x_min, motor_x_max)
        y_out[[0]] = self.map_value(y_out[[0]], 0.0, 1.0, motor_y_min, motor_y_max)
        motor_point = [x_out[[0]], y_out[[0]]]
        motor_point = np.round(motor_point).astype(int)
        motor_point = np.ravel(motor_point).tolist()
        print(motor_point)

        return motor_point

                    
    #@TODO: sustituir esto de abajo por los modelos obtenidos en la calibración.
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
    
    def pixels2stepsOld( self, point ):
        motor_point = [-1.0,-1.0] 
        if self.callibrated:
            motor_point[0] = int(self.A[0] * point[0] + self.B[0])
            motor_point[1] = int(self.A[1] * point[1] + self.B[1])        
        return motor_point
    
