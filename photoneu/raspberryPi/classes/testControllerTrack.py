from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
import sys
#sys.path.append("..")

from controller import Controller

# test para seguir un target de un color

controller = Controller()
time.sleep(3)
controller.callibrated = True

controller.set_target_color("green")

while True:
    x = controller.cam.targets[1].pos[0]
    y = controller.cam.targets[1].pos[1]
    controller.moveMotorPixels( x, y )
    time.sleep(0.3)
