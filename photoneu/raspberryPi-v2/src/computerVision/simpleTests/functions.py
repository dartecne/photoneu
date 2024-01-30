import cv2
import numpy as np
import time

window_control_name = "GUI"
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
color_dict = {'red':[0,4],'orange':[5,18],'yellow':[22,37],'green':[42,85],'blue':[92,110],'purple':[115,165],'red_2':[165,180]}  #Here is the range of H in the HSV color space represented by the color
kernel_5 = np.ones((5,5),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.

'''Hace una mascara y deja en blanco los colres muy saturado'''
def mask_colors( img ):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)              # Convert from BGR to HSV
    mask = cv2.inRange(hsv,np.array([0, 120, 120]), 
                       np.array([255, 255, 255]) )           # inRange()：Make the ones between lower/upper white, and the rest black
    dil_size = 2
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dil_size+1,2*dil_size+1),(dil_size,dil_size))
    mask = cv2.erode( mask, element )
    dil_size = 8
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dil_size+1,2*dil_size+1),(dil_size,dil_size))
    mask = cv2.dilate( mask, element )
    
    return mask

class Mice:
    """Representa todos los datos referidos a un raton"""
    light_time = 30 # segundos de tratamiento de luz
    movement_thres = 1 # umbral que se considera movimiento
    id = 0
    center = [0,0]
    center_old = [0,0]
    angle = np.angle(0.0) #radians
    color = (0,0,0) #
    duration = 0.0 # Duración del tratamiento
    init_time = time.time()
    end_time = time.time()
    ready = False
    lighting = False
    label = ""
    
    def __init__(self, id = 0, light_time = 30):
        self.duration = 0.0
        self.light_time = light_time
        self.id = id

    def update(self, position, angle, color):
        self.center_old = self.center
        self.center[0] = position[0]
        self.center[1] = position[1]
        self.angle = angle
        self.color = color
        if(self.is_stopped()): text = "stop"
        else: text = "moving"
        self.label = str(self.id) + "_" + text

    def start_light(self):
        self.init_time = time.time()
        self.lighting = True
        self.check_duration(self)

    def check_duration(self):
        now = time.time()
        duration += now - self.init_time
        if(duration >= self.light_time):
            self.ready = True
        else:
            self.ready = False
        return self.ready
#TODO no actualiza bien el valor de center_old        
    def is_stopped(self):
        print(np.linalg.norm(self.center))
        print(np.linalg.norm(self.center_old))
        if(np.linalg.norm(self.center) 
           - np.linalg.norm(self.center_old) < self.movement_thres):
            return True
        else:
            return False

class Head:
    position = (0,0)
    set_point = (0,0)
    """Send position to Arduino"""
    def move(self):
        return True
    def read_position(self):
        position = (0,0)
    def turn_on():
        print("Turn on light")
    def turn_off():
        print("Turn off light")
        
    