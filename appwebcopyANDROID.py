from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import threading
import time
import urllib.request
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)


socketio = SocketIO(app)

# En lugar de cargar un modelo YOLOv5, utiliza tu modelo entrenado en YOLOv8
model = YOLO('.\\bestv2yoloL.pt')
#url = 'http://192.168.0.100:8080/video'
#url = 'http://192.168.100.62:8080/video'
#url = 'http://192.168.137.95:8080/video'
#url = '.\\20240412_170235.mp4'
url = '.\\20240502_075656.mp4'
cap = cv2.VideoCapture(url)
def obtener_video():
    while True:
        # Leer el video desde la URL
        
        ret, frame = cap.read()
        
        # Comprobar si se ha capturado correctamente un fotograma
        if not ret:
            continue    
        frame = cv2.resize(frame, (640, 640))

        # Detección de objetos umbral de confianza
        results = model.predict(frame,conf=0.2,vid_stride=5,stream_buffer=True)#,max_det=1 .predict agregado

        # Obtener las cajas del objeto detectado
        boxes = results[0].boxes.xyxy.cpu().numpy()
        # Obtener las etiquetas de las predicciones
        class_ids = results[0].boxes.cls.cpu().numpy()
        class_names = results[0].names
        #CONFIANZA DE PREDICCION
        conf = results[0].boxes.conf.cpu().numpy()
        print(boxes)
        # Dibujar las cajas delimitadoras y las etiquetas de clase
        for box, class_id in zip(boxes, class_ids):
            
            x1, y1, x2, y2 = box[:4]
            cls = class_names[class_id]
            #if('weed' == cls):
            #    cls = 'REMOLACHA'
            #if cls == 'weed' and (x2 - x1) < 35 and (y2 - y1) < 35:
            #    cls = 'hierba'  # Cambiar el nombre de la etiqueta a "hierba" si el tamaño de la caja es pequeño
            #else:
            #    cls = 'REMOLACHA'
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(img_resized, f'{model.names[int(cls)]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, cls, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #cv2.putText(img_resized, int(conf), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Convertir el fotograma a formato .jpg
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Enviar la imagen base64 al cliente a través de SocketIO
        socketio.emit('video_feed', img_base64)
        
        # Controlar la velocidad de actualización del video
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    threading.Thread(target=obtener_video).start()   

if __name__ == '__main__':
    socketio.run(app, debug=True)
