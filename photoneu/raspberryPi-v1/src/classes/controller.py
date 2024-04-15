import numpy as np
from camHandler import CamHandler
from motorHandler import MotorHandler

#@TODO: separar en hebras el reconocimiento por visión y la escritura en el motor

class Controller:
    def __init__(self):
        self.cam = CamHandler()
        self.motor = MotorHandler()
        self.p = np.empty((2,4))    
    def callibrate(self):
        t, x_head_error, y_head_error = self.motor.getSPerror()
        while((x_head_error != 0) | (y_head_error != 0)):
            frame, hsv, gray, frame_threshold = self.cam.getImage()
            #circles = cam.findHoughCircles(frame, gray)
            n, x_cam, y_cam = self.cam.findContours(frame, frame_threshold)
            self.cam.printValues(x_cam, y_cam)
            self.cam.showImage(frame, frame_threshold)
            t, x_head_error, y_head_error = self.motor.getSPerror()
            self.motor.printValues(t, x_head_error, y_head_error)

        self.cam.printValues(x_cam, y_cam)
        t, x, y = self.motor.getMotorPosition()
#        self.p = np.append(self.p, np.random.random((1,4)), axis=0)
        self.p = np.append(self.p, [[x_cam,y_cam,x,y]], axis=0)

