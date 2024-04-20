from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
import sys
#sys.path.append("..")

from controller import Controller


# centro: 288, 227 o bien 324, 242
# up_right: 489, 94
# down_left: 127,373
# up_left: 113, 101
# down_right: 490, 357

p_cam = [[288,227], [489,94],[127,373],[113,101],[490,357]]
controller = Controller()
controller.callibrate()
for i in range(5):
    controller.testCallibration(p_cam[i][0], p_cam[i][1])
controller.cam_thread.join()
#controller.motor_thread.join()

#while True:
#    key = cv.waitKey( 30 )
#    if key == ord('q') or key == 27:
#        break