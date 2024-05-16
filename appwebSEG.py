from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import threading
import time
import urllib.request
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app)

# En lugar de cargar un modelo YOLOv5, utiliza tu modelo entrenado en YOLOv8
model = YOLO('.\\bestv1LSEG.pt')

url = 'http://192.168.137.236:8080/video' #carrito
cap = cv2.VideoCapture(url)
def obtener_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        #img_response = urllib.request.urlopen(url)
        #img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
        #img = cv2.imdecode(img_np, -1)
        # Redimensionar la imagen antes de la codificación
        #img_resized = cv2.resize(img, (640, 640))
        # Detección de objetos
        results = model(frame)
        #print(results)

        anotacion = results[0].plot()
        # Codificar la imagen con las cajas delimitadoras dibujadas en base64
       # _, img_encoded = cv2.imencode('.jpg', anotacion)
       # img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        _, img_encoded = cv2.imencode('.jpg', anotacion)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Enviar la imagen base64 al cliente a través de SocketIO
        socketio.emit('video_feed', img_base64)

        # Controla la velocidad de actualización del video
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    threading.Thread(target=obtener_video).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)

