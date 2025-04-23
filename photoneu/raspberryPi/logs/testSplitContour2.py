import cv2
import numpy as np

ruta_carpeta = r'C:\Users\inges\OneDrive - UDIT\src\photoneu\dataset\deeplabcut\labeled-data-ordered'
img_file = ruta_carpeta + r'\img0252.png'

def dividir_contorno(contorno, punto):
    # Convertimos el contorno en una lista de puntos
    contorno = contorno.squeeze()  # Elimina dimensiones extra si es necesario
    num_puntos = len(contorno)

    # Encontrar el índice del punto en el contorno
    punto = tuple(punto)  # Asegurar que el formato es correcto
    idx = np.where((contorno == punto).all(axis=1))[0]
    
    if len(idx) == 0:
        raise ValueError("El punto dado no pertenece al contorno.")
    
    idx = idx[0]  # Tomamos el primer índice encontrado

    # Rotamos la lista para que el punto dado sea el inicio
    contorno_rotado = np.roll(contorno, -idx, axis=0)

    # Dividimos el contorno en dos partes de igual tamaño
    mitad = num_puntos // 2
    parte1 = contorno_rotado[:mitad]
    parte2 = contorno_rotado[mitad:]

    return parte1, parte2

# Cargar imagen en escala de grises
imagen = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(imagen, (50, 50), (150, 150), 255, 2)

# Encontrar contornos
contornos, _ = cv2.findContours(imagen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contorno = contornos[0]  # Tomamos el primer contorno

# Seleccionar un punto del contorno manualmente
punto_inicio = tuple(contorno[10][0])  # Ejemplo: tomar el décimo punto

# Dividir el contorno
parte1, parte2 = dividir_contorno(contorno, punto_inicio)

# Dibujar las dos partes
imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
for p in parte1:
    cv2.circle(imagen_color, tuple(p), 1, (0, 255, 0), -1)  # Verde
for p in parte2:
    cv2.circle(imagen_color, tuple(p), 1, (0, 0, 255), -1)  # Rojo

# Mostrar el resultado
cv2.imshow("Contorno dividido", imagen_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
