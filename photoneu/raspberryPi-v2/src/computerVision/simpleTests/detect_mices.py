'''
TODO: Actualizar los valores de log (guardar archivos de log)
'''
from __future__ import print_function
import argparse
import cv2
import numpy as np
from array import array

from functions import *

N_MICES = 4
states = {"idle":0, "moving":1, "treatment":2, "completed":3}
state = states["idle"]

color_dict = {'red':[0,4],'orange':[5,18],'yellow':[22,37],'green':[42,85],'blue':[92,110],'purple':[115,165],'red_2':[165,180]}  #Here is the range of H in the HSV color space represented by the color
cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)

head = Head() #cabezal
mice = []
for i in range(N_MICES):
    mice.append(Mice(id = i))
cap = cv2.VideoCapture(0)
    
def end_system():
    print(">>>exiting")
#    ser.close()
    cv2.destroyAllWindows()
    exit()

# bucle de control
def loop():
    global state
# actualizar posición del cabezal
# - leer imagen
    ret, frame = cap.read()
#imagen = cv2.imread('../img/mices_color_1.png')
    imagen = frame
    assert imagen is not None, "file could not be read, check with os.path.exists()"

# - detectar mices
#imagen = cv2.medianBlur(imagen,3 ) #interesante para quitar ruido
    img_mask = mask_colors(imagen)
    _tuple = cv2.findContours(img_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)      
    # compatible with opencv3.x and openc4.x
    if len(_tuple) == 3:
        _, contours, hierarchy = _tuple
    else:
        contours, hierarchy = _tuple
    print( "Found " + str(len(contours)) + " contours" )
    #cv2.drawContours( imagen, contours, -1, ( 0, 255, 0 ), 3 )

# - actualizar valores de los ratones
    id = 0
    for cnt in contours:
        (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
        (x,y) = map(int, (x,y))
        cnt_color = array("i", imagen[y,x])
        mice[id].update((x,y), 0.0, cnt_color)
   
        cv2.putText(imagen,mice[id].label,
            (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,cnt_color,2)
        id+=1
        if(id >= N_MICES):
            break;  

# - actualizar cabezal según datos del mice
#     - mover el cabezal hasta llegar al lugar del ratón
   
    cv2.imshow( window_capture_name, imagen )
    cv2.imshow( "Mask", img_mask )
# - actualizar valores de tratamiento de luz al ratón
    id = -1
    if(state == states["idle"]):
    #buscamos el primer ratón que esté quieto
        for m in mice:
            if( m.is_stopped() & m.ready != True ):
                head.set_point = m.center
                head.move()
                id = m.id
                state=states["moving"]
                break
    head.read_position()    
    if( head.position == head.set_point ):
        if( state==states["moving"] ):
            state=states["treatment"]
            head.turn_on()
            mice[i].start_light()
    if state == states["treatment"]:
        mice[i].check_duration()
        if mice[i].ready:
            head.turn_off()
            state=states["idle"]
        
    if all([m.ready for m in mice]):
        state = states["completed"]
    key = cv2.waitKey(30)

    if key == ord('q') or key == 27:
        end_system()
        
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

while True:
    loop()
    time.sleep(0.1)

#key = cv2.waitKey(30)
#if key == ord('q') or key == 27:
#    cv2.destroyAllWindows()
#    exit()
