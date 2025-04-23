import cv2
import numpy as np

def dividir_contorno(contorno, punto):
    # Convertir contorno a lista de tuplas para facilitar la búsqueda
    contorno = contorno.squeeze()  # Eliminar dimensiones extra si existen
    if(len(contorno)< 5):
        return None, None
    contorno_lista = contorno.tolist()
    punto = list(punto)  # Asegurar que el formato es correcto
    print("punto:", punto)
    print("Contorno:", contorno_lista)
    

    idx = contorno_lista.index(punto)
    
    # Determinar el tamaño de las mitades
    n = len(contorno)
    mitad = n // 2

    if idx < mitad:
        parte1 = contorno[idx:idx + mitad]
        parte2 = np.concatenate((contorno[idx + mitad:], contorno[:idx]))
    else:
        parte1 = np.concatenate((contorno[idx:], contorno[:idx - mitad]))
        parte2 = contorno[idx - mitad:idx]
    # Dividir el contorno en dos partes de igual tamaño
#    parte1 = contorno_lista[indice_punto:indice_punto + mitad]
#    parte2 = contorno_lista[indice_punto + mitad:] + contorno_lista[:indice_punto]

    return np.array(parte1), np.array(parte2)

ruta_carpeta = r'C:\Users\inges\OneDrive - UDIT\src\photoneu\dataset\deeplabcut\labeled-data-ordered'
img_file = ruta_carpeta + r'\img0252.png'
# Cargar imagen y extraer contornos
imagen = cv2.imread(img_file)
imagen_color = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
_, umbral = cv2.threshold(imagen_color, 57, 255, cv2.THRESH_BINARY_INV)
contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(len(contornos))
for contorno in contornos:
    print(str(len(contorno)))
    if(len(contorno) < 5 ):
        continue
    punto = tuple(contorno[len(contorno)- len(contorno)//4][0])  # Elegir un punto de prueba
#    punto = tuple(contorno[len(contorno)//2][0])  # Elegir un punto de prueba

    parte1, parte2 = dividir_contorno(contorno, punto)

    # Dibujar los segmentos en colores distintos

    if parte1 is not None and parte2 is not None:
        cv2.polylines(imagen, [parte1], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.polylines(imagen, [parte2], isClosed=False, color=(0, 0, 255), thickness=2)
#    cv2.polylines(imagen_color, [contorno], isClosed=False, color=(255, 0, 0), thickness=2)

cv2.imshow('Contorno dividido', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
