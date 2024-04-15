from __future__ import print_function
import argparse
import time
import cv2 as cv

from camHandler import CamHandler

cam = CamHandler()

while True:
    frame, hsv, gray, frame_threshold = cam.getImage()
    #circles = cam.findHoughCircles(frame, gray)
    n, x, y = cam.findContours(frame, frame_threshold)
    cam.showImage(frame, frame_threshold)

    key = cv.waitKey( 30 )
    if key == ord('q') or key == 27:
        break
    