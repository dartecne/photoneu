import cv2
import numpy as np

def detectar_rojo(imagen):
    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir el rango de colores rojos en HSV
    rango_bajo = np.array([0, 100, 100])
    rango_alto = np.array([10, 255, 255])

    # Crear una máscara utilizando el rango de colores rojos
    mascara1 = cv2.inRange(hsv, rango_bajo, rango_alto)

    # Definir un segundo rango de colores rojos en HSV
    rango_bajo = np.array([160, 100, 100])
    rango_alto = np.array([179, 255, 255])

    # Crear otra máscara utilizando el segundo rango de colores rojos
    mascara2 = cv2.inRange(hsv, rango_bajo, rango_alto)

    # Combinar las dos máscaras
    mascara_roja = cv2.bitwise_or(mascara1, mascara2)

    # Aplicar la máscara a la imagen original
    resultado = cv2.bitwise_and(imagen, imagen, mask=mascara_roja)

    return resultado

# Cargar la imagen
imagen = cv2.imread('../img/mices_color_1.png')

# Llamar a la función para detectar zonas de color rojo
resultado = detectar_rojo(imagen)

# Mostrar la imagen original y la imagen con las zonas de color rojo resaltadas
cv2.imshow('Imagen Original', imagen)
cv2.imshow('Zonas de Color Rojo', resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()