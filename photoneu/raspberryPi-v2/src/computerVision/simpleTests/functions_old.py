import cv2
import numpy as np

percent = 1
percent_name = "Porcentaje histograma"
window_control_name = "GUI"
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
color_dict = {'red':[0,4],'orange':[5,18],'yellow':[22,37],'green':[42,85],'blue':[92,110],'purple':[115,165],'red_2':[165,180]}  #Here is the range of H in the HSV color space represented by the color
kernel_5 = np.ones((5,5),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.

def on_percent_trackbar(val):
    global percent
    percent = val
    cv2.setTrackbarPos(percent_name, window_control_name, percent)

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

def color_detect( img, color_name ):

    # The blue range will be different under different lighting conditions and can be adjusted flexibly.  H: chroma, S: saturation v: lightness
#    resize_img = cv2.resize(img, (160,120), interpolation=cv2.INTER_LINEAR)  # In order to reduce the amount of calculation, the size of the picture is reduced to (160,120)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)              # Convert from BGR to HSV
    
    mask = cv2.inRange(hsv,np.array([min(color_dict[color_name]), 200, 200]), 
                       np.array([max(color_dict[color_name]), 255, 255]) )           # inRange()：Make the ones between lower/upper white, and the rest black
    if color_name == 'red':
            mask_2 = cv2.inRange(hsv, (color_dict['red_2'][0],0,0), (color_dict['red_2'][1],255,255)) 
            mask = cv2.bitwise_or(mask, mask_2)

    morphologyEx_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5,iterations=1)              # Perform an open operation on the image 

    # Find the contour in morphologyEx_img, and the contours are arranged according to the area from small to large.
    _tuple = cv2.findContours(morphologyEx_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)      
    # compatible with opencv3.x and openc4.x
    if len(_tuple) == 3:
        _, contours, hierarchy = _tuple
    else:
        contours, hierarchy = _tuple
    
    #color_area_num = len(contours) # Count the number of contours
    #print("color areas found: " + str(color_area_num))
    #areas = []
    #i = 0
    #if color_area_num > 0: 
    #    for i in contours:    # Traverse all contours
    #        x,y,w,h = cv2.boundingRect(i)      # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition object

            # Draw a rectangle on the image (picture, upper left corner coordinate, lower right corner coordinate, color, line width)
#            if w >= 2 and h >= 2: # Because the picture is reduced to a quarter of the original size, if you want to draw a rectangle on the original picture to circle the target, you have to multiply x, y, w, h by 4.
    #        if True: # Because the picture is reduced to a quarter of the original size, if you want to draw a rectangle on the original picture to circle the target, you have to multiply x, y, w, h by 4.
    #            x = x * 4
    #            y = y * 4 
    #            w = w * 4
    #            h = h * 4
    #            cv2.rectangle(img,(x,y),(x+20,y+20),(0,0,255),2)  # Draw a rectangular frame
    #            cv2.putText(img,color_name,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)# Add character description

    return img, mask, morphologyEx_img, contours

def detectar_colores_sobresalientes(imagen, porcentaje=1):
    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Calcular el histograma de la saturación
    hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])

    # Obtener los valores de saturación más altos
    porcentaje_pixels = int(np.sum(hist) * (porcentaje / 1000.0))
    valores_saturacion = np.where(np.cumsum(hist) >= porcentaje_pixels)[0][0]

    # Definir el rango de colores basado en la saturación
    rango_bajo = np.array([0, valores_saturacion, 0])
    rango_alto = np.array([180, 255, 255])

    # Crear una máscara utilizando el rango de colores basado en la saturación
    mascara = cv2.inRange(hsv, rango_bajo, rango_alto)

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)
    img_colors = cv2.cvtColor(resultado, cv2.COLOR_BGR2HSV)
#    resultado = cv2.cvtColor(resultado, cv2.COLOR_HSV2BGR)

    return resultado, img_colors, hist

def control_loop():
    # path from raspberryPi-v2
  imagen = cv2.imread('./src/computerVision/img/mices_color_1.png')
#  img_colors, hsv_values = detectar_colores_sobresalientes(imagen, percent)
#  color_contours = 
#  for color_id in color_dict:
  color_id = "green"
  if True:
    img_colors, img_mask, img_morph, contours = color_detect(imagen, color_id)
    cnt = max(contours, key = cv2.contourArea)
    print(str(color_id) + " - max area = " + str(cv2.contourArea(cnt)))
    e = cv2.fitEllipse(cnt) #[0]=center, [1]-radius, [2]-angle
    (x,y) = map(int, e[0]) 
    # x,y = ellipse.center
    ellipse_color = (max(color_dict[color_id]), 255, 255)
    cv2.ellipse(imagen,e,ellipse_color,2)
    cv2.putText(imagen,color_id,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)# Add character description


#  cv2.drawContours( img_colors, contours, -1, (0,255,0), 3  )
  
#   h,w,c = img_colors.shape
#   scale = 0.5
#   img_colors_resize = cv2.resize(img_colors,None, fx = scale, fy = scale, interpolation=cv2.INTER_LINEAR)
#   img_colors_resize = cv2.cvtColor(img_colors_resize, cv2.COLOR_BGR2GRAY)
#   img_colors_resize = cv2.blur(img_colors_resize, (3,3))
#   kernel_5 = np.ones((3,3),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.
#   frame_threshold = cv2.morphologyEx( img_colors_resize, cv2.MORPH_OPEN, kernel_5,iterations=1 )
#   _tuple = cv2.findContours(frame_threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)      
  
#   if len(_tuple) == 3:
#     _, contours, hierarchy = _tuple
#   else:
#     contours, hierarchy = _tuple

#   color_area_num = len(contours) # Count the number of contours

#   if color_area_num > 0: 
#     for i in contours:    # Traverse all contours
#         x,y,w,h = cv2.boundingRect(i)      # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition object

# #        if w >= 8 and h >= 8: #
#         if True: 
#           cv2.rectangle(img_colors_resize,(x,y),(x+w,y+h),(120,20,200),2)  # Draw a rectangular frame
# #        if( (w * h > minMicePixelArea) & (w * h < maxMicePixelArea)):
#           text = "mice"# + str(contours.index(i))
#           pointText = str(x) + ", " + str(y)
#           cv2.putText(img_colors_resize, text,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1)# Add character description
#           cv2.putText(img_colors_resize, pointText ,(x,y+40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),1)# Add character description

  cv2.imshow(window_capture_name, img_colors)
  cv2.imshow("Mask", img_mask)
  cv2.imshow("Morph", img_morph)

  key = cv2.waitKey(30)
  if key == ord('q') or key == 27:
    cv2.destroyAllWindows()
    exit()

