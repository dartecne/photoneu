from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import time
import sys
#sys.path.append("..")

from controller import Controller

# Test para realizar la calibración de la máquina

#   posiciones del header en pixeles
# centro: 288, 227 o bien 324, 242
# up_right: 489, 94
# down_left: 127,373
# up_left: 113, 101
# down_right: 490, 357

p_cam = [[220,150], [100,94], [100,277], [250,270], [250,100]]
controller = Controller()
input("pulse una tecla para comenzar con la calibración")
controller.calibrate()
#file_name = "2025_04_29_11_06_11.log"
#controller.analyze_calibration_data(file_name)
input("pulse una tecla para finalizar")
controller.endSystem()
#input("pulse una tecla para mover el cabezal")

#for i in range(5):
#    controller.moveMotorPixels( p_cam[i][0], p_cam[i][1] )
#    time.sleep(1)
#controller.set_target_color('blue')
