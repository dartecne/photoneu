from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
import sys
#sys.path.append("..")

from controller import Controller


#   posiciones del header en pixeles
# centro: 288, 227 o bien 324, 242
# up_right: 489, 94
# down_left: 127,373
# up_left: 113, 101
# down_right: 490, 357

p_cam = [[220,150], [78,64], [78,277], [380,270], [384,64]]
controller = Controller()
time.sleep(9)
#controller.callibrated = True
controller.callibrate()
for i in range(5):
    controller.moveMotorPixels( p_cam[i][0], p_cam[i][1] )
    time.sleep(1)
#controller.color = 'blue'
while True:
    while controller.cam.target.is_moving == False:
        x = controller.cam.target.pos[0]
        y = controller.cam.target.pos[1]
        controller.moveMotorPixels( x, y )

#controller.cam.join()
#controller.motor_thread.join()

#while True:
#    key = cv.waitKey( 30 )
#    if key == ord('q') or key == 27:
#        break