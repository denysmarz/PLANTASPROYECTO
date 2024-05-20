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
model = YOLO('.\\best.pt')

url = 'http://192.168.137.201/cam-hi.jpg' #carrito
# Dirección IP y puerto del ESP32
ESP32_CAM_IP = '192.168.137.201'

def obtener_video():
    while True:
        img_response = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        # Redimensionar la imagen antes de la codificación
        img_resized = cv2.resize(img, (640, 640))
        
        # Detección de objetos

        """"
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

        """
        # Codificar la imagen con las cajas delimitadoras dibujadas en base64
        _, img_encoded = cv2.imencode('.jpg', img_resized)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Enviar la imagen base64 al cliente a través de SocketIO
        socketio.emit('video_feed', img_base64)

        # Controla la velocidad de actualización del video
        time.sleep(0.1)

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


@socketio.on('connect')
def handle_connect():
    threading.Thread(target=obtener_video).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)

