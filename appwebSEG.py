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

model = YOLO('.\\bestv1LSEG.pt')
modelHUMEDAD = YOLO('.\\bestv2HUMEDAD.pt')

#url = 'http://192.168.137.236:8080/video' #carrito
url = '.\\20240423_132425.mp4'
cap = cv2.VideoCapture(url)
def obtener_video():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Detección de objetos

        frame = cv2.resize(frame, (640, 640))
        results = model.predict(frame,conf=0.5)

        anotacion = results[0].plot()

        boxes = results[0].boxes.xyxy.cpu().numpy()
        #falta verificar en caso de ninguna detenccion---------------
        #falta mejorar la interfas
        #falta la parte robotica--
        #se usara mysql?
        #se ara una verificacion de ultimo riego 
        #se usara cantidad de agua exacta? o hasta que modelo detecte humedo?
        #mas fotos
        #falta documento
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Recortar la imagen utilizando las coordenadas
            cropped_image = frame[y1:y2, x1:x2]

            # Realizar alguna operación con el modelo secundario
            #cropped_image_resized = cv2.resize(cropped_image, (640, 640))  # Redimensionar si es necesario
            other_results = modelHUMEDAD.predict(cropped_image)
            #class_ids = other_results[0].boxes.cls.cpu().numpy()
            probs = other_results[0].probs.data.cpu().numpy()
            print(probs[0])#probs[0] HUMEDAD - probs[1] SEQUEDAD
            cv2.putText(anotacion, str(round(probs[0],3)), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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

