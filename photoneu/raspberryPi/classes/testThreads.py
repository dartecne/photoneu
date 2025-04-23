import time
import threading
import numpy as np
import cv2 as cv
import random

from camHandler import CamHandler
from motorHandler import MotorHandler


cam = CamHandler()
motor = MotorHandler()

def cam_thread(name):
    while True:
        frame, hsv, gray = cam.getImage()
        frame_threshold = cam.filterColor(hsv, 'red')
        #circles = cam.findHoughCircles(frame, gray)
        n, x, y = cam.findContours(frame, frame_threshold)
        cam.showImage(frame, frame_threshold)

        key = cv.waitKey( 30 )
        if key == ord('q') or key == 27:
            break
 

def motor_thread(name):
    i = 0
    time.sleep(9)
    while True:
        print("position = ")
        t, x, y = motor.getMotorPosition()
        print( str(x) + ", " + str(y) )
        print("error = ") 
        print( motor.getSPerror() )
        dx = random.randrange(-2000,2000,1)
        dy = random.randrange(-2000,2000,1)
        motor.moveHead( x + dx, y + dy )
#        print(str(name) + str(i))
        i += 1
        time.sleep( 2 )
        key = cv.waitKey( 30 )
        if key == ord('q') or key == 27:
            break

if __name__ == "__main__":
    x = threading.Thread( target = cam_thread, args=(1,) )
    y = threading.Thread( target = motor_thread, args=(1,) )
    x.start()
    y.start()
    x.join()
    y.join()