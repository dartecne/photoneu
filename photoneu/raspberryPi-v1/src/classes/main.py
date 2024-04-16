from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
import sys
#sys.path.append("..")

from controller import Controller

controller = Controller()
time.sleep(7)
print("main- callibrating...")
controller.callibrate()
time.sleep(1)
print("main- test callibration...")
controller.testCallibration()
