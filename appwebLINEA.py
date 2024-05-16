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

# En lugar de cargar un modelo YOLOv5, utiliza tu modelo entrenado en YOLOv8
model = YOLO('.\\bestNANO100.pt')
# Dirección IP y puerto del ESP32
ESP32_CAM_IP = '192.168.137.208'
url = 'http://192.168.137.208/cam-hi.jpg' #plantas
url2 = 'http://192.168.34.199:8080/video' #carrito
#cap = cv2.VideoCapture(url2)

def obtener_video():
    seguir_linea()
    """while True:

        img_response = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        # Redimensionar la imagen antes de la codificación
        img_resized = cv2.resize(img, (640, 640))
        # Detección de objetos
        results = model(img_resized)
        # Obtener las cajas del objeto detectado
        boxes = results[0].boxes.xyxy.cpu().numpy()
        # Obtener las etiquetas de las predicciones
        class_ids = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names
    
        # Dibujar las cajas delimitadoras y las etiquetas de clase
        for box, class_id in zip(boxes, class_ids):
            x1, y1, x2, y2 = box[:4]
            cls = class_names[class_id]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(img_resized, f'{model.names[int(cls)]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img_resized, cls, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Codificar la imagen con las cajas delimitadoras dibujadas en base64
        _, img_encoded = cv2.imencode('.jpg', img_resized)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Enviar la imagen base64 al cliente a través de SocketIO
        socketio.emit('video_feed', img_base64)

        # Controla la velocidad de actualización del video
        time.sleep(0.1)"""

@app.route('/')
def index():
    return render_template('index copy.html')

@app.route('/control-motores/<int:motor1>/<int:motor2>')
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

def seguir_linea():
    
    while True:
        cap = cv2.VideoCapture(url2)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (640, 640))
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
            cv2.line(frame, leftmost, rightmost, (0, 255, 255), 10)
            
            # Tomar decisiones para controlar el coche
            center_x = frame.shape[1] // 2
            line_center_x = (leftmost[0] + rightmost[0]) // 2
            if line_center_x < center_x - 20:  # Si la línea está a la izquierda del centro
                control_motores(5, 1)  # Gira a la izquierda
            elif line_center_x > center_x + 20:  # Si la línea está a la derecha del centro
                control_motores(4, 1 )  # Gira a la derecha
            else:
                control_motores(1, 1)  # Avanza
        else:
            control_motores(2, 1)  # Frena si no se detecta ninguna línea
        
        # Mostrar el fotograma con la línea amarilla dibujada
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        socketio.emit('video_feed', img_base64)

@socketio.on('connect')
def handle_connect():
    threading.Thread(target=obtener_video).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)

