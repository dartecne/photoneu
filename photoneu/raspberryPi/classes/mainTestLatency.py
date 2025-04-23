from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
import sys
#sys.path.append("..")

from controller import Controller

# Se mueve 2000 steps de forma radial y va guardando los tiempos
# tmotor (comienza), tsp (llega al SP), tcam(camara lo detecta)
controller = Controller()
time.sleep(3)

controller.latencyTest()

while True:
    key = cv.waitKey( 30 )
    if key == ord('q') or key == 27:
        break