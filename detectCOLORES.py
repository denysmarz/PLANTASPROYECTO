import cv2
import numpy as np

def find_and_draw_contours(mask, frame, color, line_position_h, line_position_v, area_threshold=1000):
    # Encontrar contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar cajas delimitadoras alrededor de los contornos encontrados
    for contour in contours:
        # Calcular el área del contorno
        area = cv2.contourArea(contour)
        
        # Obtener las coordenadas de la caja delimitadora
        x, y, w, h = cv2.boundingRect(contour)
        
        # Ignorar contornos con un área pequeña, que estén por encima de la línea horizontal
        # y que estén a la derecha de la línea vertical
        if area > area_threshold and y + h > line_position_h and x < line_position_v:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

# Capturar video desde la cámara
cap = cv2.VideoCapture('.\\20240417_102015.mp4')

# Variable de control para pausar/reanudar el video
pausado = False

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convertir el frame a espacio de color HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Definir el rango de color azul en HSV
    azul_bajo = np.array([100, 100, 100])
    azul_alto = np.array([140, 255, 255])
    
    # Definir el rango de color para las hojas de remolacha en HSV
    hojas_remolacha_bajo = np.array([25, 50, 50])
    hojas_remolacha_alto = np.array([90, 255, 255])
    
    # Crear máscaras para los colores azul y de las hojas de remolacha
    mask_azul = cv2.inRange(hsv, azul_bajo, azul_alto)
    mask_remolacha = cv2.inRange(hsv, hojas_remolacha_bajo, hojas_remolacha_alto)
    
    # Combinar las máscaras
    mask_combined = cv2.bitwise_or(mask_azul, mask_remolacha)
    
    # Obtener las dimensiones del frame
    height, width, _ = frame.shape

    # Dibujar una línea horizontal y una línea vertical en el centro del frame
    # Calcular la posición vertical deseada para la línea horizontal
    line_position_h = height // 2 - 70  # Ajuste la cantidad según sea necesario
    # Calcular la posición horizontal deseada para la línea vertical
    line_position_v = width // 2

    # Dibujar la línea horizontal y vertical azul en las posiciones deseadas
    cv2.line(frame, (0, line_position_h), (width, line_position_h), (255, 0, 0), 3)  # Línea horizontal
    cv2.line(frame, (line_position_v, 0), (line_position_v, height), (255, 0, 0), 3)  # Línea vertical
    
    # Encontrar contornos y dibujar cajas delimitadoras
    find_and_draw_contours(mask_combined, frame, (0, 255, 0), line_position_h, line_position_v, area_threshold=1000)  # Verde para las cajas delimitadoras
    
    # Mostrar el frame con las cajas delimitadoras
    cv2.imshow('Detected Objects', frame)
    
    # Pausar/reanudar el video cuando se presiona la tecla de espacio
    key = cv2.waitKey(1)
    if key == 32:  # 32 es el código de la tecla de espacio
        pausado = not pausado  # Cambiar el estado de pausa
    if pausado:
        key = cv2.waitKey(0)  # Esperar hasta que se presione una tecla (cualquier tecla)
        pausado = False  # Reanudar el video
    
    # Romper el bucle si se presiona la tecla 'q'
    if key & 0xFF == ord('q'):
        break

# Liberar el objeto VideoCapture y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
