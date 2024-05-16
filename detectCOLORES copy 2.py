from flask import Flask, render_template,request
from flask_socketio import SocketIO
import cv2
import base64
import threading
import time
import urllib.request
import numpy as np
from ultralytics import YOLO
import requests

app = Flask(__name__)
socketio = SocketIO(app)

ESP32_CAM_IP = '192.168.137.98'
url2 = 'http://192.168.137.95:8080/video'
cap = cv2.VideoCapture(url2)
def seguir_linea():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Aplicar operaciones de procesamiento de imágenes para detectar la línea amarilla
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Encontrar contornos de la línea amarilla
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si se detecta al menos un contorno
        if contours:
            # Encontrar el contorno más grande (la línea amarilla)
            largest_contour = max(contours, key=cv2.contourArea)
            # Obtener los extremos izquierdo y derecho de la línea amarilla
            leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
            # Dibujar una línea sobre la línea amarilla
            cv2.line(frame, leftmost, rightmost, (0, 255, 255), 2)
            
            # Tomar decisiones para controlar el coche
            center_x = frame.shape[1] // 2
            line_center_x = (leftmost[0] + rightmost[0]) // 2
            if line_center_x < center_x - 20:  # Si la línea está a la izquierda del centro
                control_motores(1, 5)  # Gira a la izquierda
            elif line_center_x > center_x + 20:  # Si la línea está a la derecha del centro
                control_motores(1, 4)  # Gira a la derecha
            else:
                control_motores(1, 1)  # Avanza
        else:
            control_motores(1, 2)  # Frena si no se detecta ninguna línea
        
        # Mostrar el fotograma con la línea amarilla dibujada
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        socketio.emit('video_feed', img_base64)

def control_motores(motor1, motor2):
    # Construir la URL para enviar la solicitud al ESP32-CAM
    url = f'http://{ESP32_CAM_IP}/control-motores?motor1={motor1}&motor2={motor2}'
    # Enviar la solicitud al ESP32-CAM
    response = requests.get(url)
    
    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        return 'Comandos enviados al ESP32-CAM exitosamente'
    else:
        return 'Error al enviar comandos al ESP32-CAM'
