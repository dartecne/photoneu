from picamera import PiCamera
from time import sleep

camera = PiCamera()

#camera.start_preview()
#camera.start_preview(alpha=200)
camera.start_preview()
sleep(12)
camera.stop_preview()
