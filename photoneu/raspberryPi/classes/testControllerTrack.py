from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
import sys
#sys.path.append("..")

from controller import Controller

# test para seguir un target de un color

controller = Controller(True)
time.sleep(3)

while True:
    x = controller.cam.targets[0].pos[0]
    y = controller.cam.targets[0].pos[1]
    controller.moveMotorPixels( x, y )
    time.sleep(0.3)
