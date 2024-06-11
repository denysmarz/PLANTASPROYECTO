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
import requests
#falta verificar en caso de ninguna detenccion---------------
        #falta mejorar la interfas
        #falta la parte robotica--
        #se usara mysql?
        #se ara una verificacion de ultimo riego 
        #se usara cantidad de agua exacta? o hasta que modelo detecte humedo?
        #mas fotos
        #falta documento
app = Flask(__name__)
socketio = SocketIO(app)

model = YOLO('.\\bestSEGMENTACIONv3originalesaumento.pt')
modelHUMEDAD = YOLO('.\\bestHUMEDADv3PRE.pt')

#url = 'http://192.168.137.236:8080/video' #carrito
url = '.\\20240531_174518.mp4'
#'.\\20240519_173228.mp4','.\\20240519_173140.mp4','.\\20240519_173059.mp4','.\\20240502_075656.mp4'
#ESP32
url2 = 'http://192.168.137.110/cam-hi.jpg ' #camara esp32
# Dirección IP y puerto del ESP32
ESP32_CAM_IP = '192.168.137.110'
cap = cv2.VideoCapture(url)

analysis_started = False

def obtener_video():
    global analysis_started
    controlador = 0
    while analysis_started == True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Detección de objetos

        frame = cv2.resize(frame, (640, 640))
        results = model.predict(frame,conf=0.3)

        anotacion = results[0].plot()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        humidity_prob = 0
        cantidad_detecciones = 0
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
            humidity_prob = round(probs[0], 3)+humidity_prob
            cantidad_detecciones = 1 + cantidad_detecciones
            #print(humidity_prob)#probs[0] HUMEDAD - probs[1] SEQUEDAD
            cv2.putText(anotacion, str(round(probs[0],3)), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        
            print("HUMEDAD TOTAL = ",humidity_prob)
        print("CANTIDAD DE DETECCIONES = ",cantidad_detecciones)
        if cantidad_detecciones != 0:
            humidity_promedio = humidity_prob/cantidad_detecciones
        _, img_encoded = cv2.imencode('.jpg', anotacion)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Enviar la imagen base64 al cliente a través de SocketIO
         # Emitir la imagen base64 y la probabilidad de humedad al cliente
        
        controlador +=1
        if(controlador == 3):
            if(humidity_promedio < 0.3):
                #mover_motor_regador(x1, frame.shape[1])
                #abrir_llave_de_paso()
                print("RIEGO ACTIVADO,HUMEDAD = ",humidity_promedio)
            analysis_started = False
            socketio.emit('video_feed', {'img': img_base64, 'humidity_prob': round(humidity_promedio, 3),'mensaje': "ANALISIS TERMINADO"})
            print("ANALISIS TERMINADO")
        else:
            socketio.emit('video_feed', {'img': img_base64, 'humidity_prob': 0,'mensaje':"ANALIZANDO"})
        # Controla la velocidad de actualización del video
        time.sleep(0.1)

def mover_motor_regador(posicion_x, ancho_frame):
    centro_frame = ancho_frame // 2
    if posicion_x < centro_frame:
        direccion = 1  # Mover hacia la izquierda
    else:
        direccion = 2  # Mover hacia la derecha

    url = f'http://{ESP32_CAM_IP}/control-motores?motor1={direccion}&motor2=0'
    response = requests.get(url)
    if response.status_code == 200:
        print('Comando de movimiento enviado al ESP32-CAM exitosamente')
    else:
        print('Error al enviar el comando de movimiento al ESP32-CAM')

def abrir_llave_de_paso():
    # Abre la llave de paso
    url = f'http://{ESP32_CAM_IP}/control-motores?motor1=3&motor2=0'
    response = requests.get(url)
    if response.status_code == 200:
        print('Llave de paso abierta a la mitad')
    else:
        print('Error al abrir la llave de paso')

    # Esperar 30 segundos para abrir la llave de paso completamente
    time.sleep(30)
    url = f'http://{ESP32_CAM_IP}/control-motores?motor1=4&motor2=0'
    response = requests.get(url)
    if response.status_code == 200:
        print('Llave de paso abierta completamente')
    else:
        print('Error al abrir completamente la llave de paso')

    # Cerrar la llave de paso después de 1 minuto
    threading.Timer(60, cerrar_llave_de_paso).start()

def cerrar_llave_de_paso():
    url = f'http://{ESP32_CAM_IP}/control-motores?motor1=4&motor2=0'
    response = requests.get(url)
    if response.status_code == 200:
        print('Llave de paso cerrada')
    else:
        print('Error al cerrar la llave de paso')

@app.route('/')
def index():
    return render_template('index copy 2.html')

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

@socketio.on('start_analysis')
def handle_start_analysis():
    print('VERIFICANDO CUADRANTE')

    global analysis_started



    



    analysis_started = True
    print('Análisis de video iniciado')
    threading.Thread(target=obtener_video).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)

