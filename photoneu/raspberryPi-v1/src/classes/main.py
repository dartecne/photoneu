from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
import sys
#sys.path.append("..")

from controller import Controller

controller = Controller()
controller.callibrate()