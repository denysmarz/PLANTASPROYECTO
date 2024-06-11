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
from datetime import datetime
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
url = '.\\20240519_173140.mp4'
#'.\\20240519_173228.mp4','.\\20240519_173140.mp4','.\\20240519_173059.mp4','.\\20240502_075656.mp4','.\\20240519_173114.mp4'seco
#ESP32
url2 = 'http://192.168.137.110/cam-hi.jpg ' #camara esp32
# Dirección IP y puerto del ESP32
ESP32_CAM_IP = '192.168.137.110'
cap = cv2.VideoCapture(url)

analysis_started = False
position_message = ""
verificador_de_riego = ["no regado","no regado","no regado"]#0 = derecha, 1 = centro, 2 = izquierda

def obtener_video():
    global analysis_started
    global position_message
    global verificador_de_riego
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
                analysis_started = False
                socketio.emit('video_feed', {'img': img_base64, 'humidity_prob': round(humidity_promedio, 3),'mensaje': "ANALISIS TERMINADO"})
                print("RIEGO ACTIVADO,HUMEDAD = ",humidity_promedio)
                socketio.emit('activar_riego', {'activar_r': "Humedad Baja iniciando riego"})
                mover_motor_regador(position_message)
            else:
                if position_message == "Pelota detectada a la derecha":
                    verificador_de_riego[0] = "Verificado"
                elif position_message == "Pelota detectada en el centro":
                    verificador_de_riego[1] = "Verificado"
                elif position_message == "Pelota detectada a la izquierda":
                    verificador_de_riego[2] = "Verificado"
            
            analysis_started = False
            socketio.emit('video_feed', {'img': img_base64, 'humidity_prob': round(humidity_promedio, 3),'mensaje': "ANALISIS TERMINADO"})
            print("ANALISIS TERMINADO")
            socketio.emit('activar_riego', {'activar_r': ""})
            time.sleep(20)
            if verificador_de_riego[0] == "regado" and verificador_de_riego[1] == "regado" and verificador_de_riego[2] == "regado":
                print("TODOS LOS CAMPOS FUERON REGADOS")
                #registrar fecha y horadel riego
                socketio.emit('activar_riego', {'activar_r': "Todos los campos fueron regados"})
                now = datetime.now()
                print(now)
                break
            else:    
                handle_start_analysis()

        else:
            socketio.emit('video_feed', {'img': img_base64, 'humidity_prob': 0,'mensaje':"ANALIZANDO"})
        # Controla la velocidad de actualización del video
        time.sleep(0.1)

def mover_motor_regador(posicion_a_ir):
    if posicion_a_ir == "Pelota detectada a la derecha":
        contador = 0 
        while contador < 2:
            url = f'http://{ESP32_CAM_IP}/control-motores?motor1=1&motor2=0'
            contador +=1
            response = requests.get(url)
            if response.status_code == 200:
                print('Comando de movimiento enviado al ESP32-CAM exitosamente')
            else:
                print('Error al enviar el comando de movimiento al ESP32-CAM')
            time.sleep(1.1)
        #abrir llave de paso
        abrir_llave_de_paso()

    if posicion_a_ir == "Pelota detectada en el centro":
        abrir_llave_de_paso()

    if posicion_a_ir == "regresar de derecha":
        contador = 0 
        while contador < 2:
            url = f'http://{ESP32_CAM_IP}/control-motores?motor1=2&motor2=0'
            contador +=1
            response = requests.get(url)
            if response.status_code == 200:
                print('Comando de movimiento enviado al ESP32-CAM exitosamente')
            else:
                print('Error al enviar el comando de movimiento al ESP32-CAM')
            time.sleep(1.1)

    if posicion_a_ir == "Pelota detectada a la izquierda":
        contador = 0 
        while contador < 2:
            url = f'http://{ESP32_CAM_IP}/control-motores?motor1=2&motor2=0'
            contador +=1
            response = requests.get(url)
            if response.status_code == 200:
                print('Comando de movimiento enviado al ESP32-CAM exitosamente')
            else:
                print('Error al enviar el comando de movimiento al ESP32-CAM')
            time.sleep(1.1)
        #abrir llave de paso
        abrir_llave_de_paso()

    if posicion_a_ir == "regresar de izquierda":
        contador = 0 
        while contador < 2:
            url = f'http://{ESP32_CAM_IP}/control-motores?motor1=1&motor2=0'
            contador +=1
            response = requests.get(url)
            if response.status_code == 200:
                print('Comando de movimiento enviado al ESP32-CAM exitosamente')
            else:
                print('Error al enviar el comando de movimiento al ESP32-CAM')
            time.sleep(1.1)

def abrir_llave_de_paso():
    # Abre la llave de paso
    contador = 0 
    while contador < 4:
        url = f'http://{ESP32_CAM_IP}/control-motores?motor1=3&motor2=0'
        contador +=1
        response = requests.get(url)
        if response.status_code == 200:
            print('Comando de movimiento enviado al ESP32-CAM exitosamente')
        else:
            print('Error al enviar el comando de movimiento al ESP32-CAM')
        time.sleep(1.1)
    #una vez abierto domir 30 segundos despues cerrar el agua    
    #time.sleep(10)
    # Cerrar la llave de paso después de 10 segundos
    threading.Timer(10, cerrar_llave_de_paso).start()

def cerrar_llave_de_paso():
    global position_message
    global verificador_de_riego
    contador = 0 
    while contador < 4:
        url = f'http://{ESP32_CAM_IP}/control-motores?motor1=4&motor2=0'
        contador +=1
        response = requests.get(url)
        if response.status_code == 200:
            print('Comando de movimiento enviado al ESP32-CAM exitosamente')
        else:
            print('Error al enviar el comando de movimiento al ESP32-CAM')
        time.sleep(1.1)

    if position_message == "Pelota detectada a la derecha":
        verificador_de_riego[0] = "regado"
        position_message = "regresar de derecha"
    if position_message == "Pelota detectada en el centro":
        verificador_de_riego[1] = "regado"
        position_message = "estamos en el centro"
    if position_message == "Pelota detectada a la izquierda":
        verificador_de_riego[2] = "regado"
        position_message = "regresar de izquierda"
    mover_motor_regador(position_message)

@app.route('/')
def index():
    return render_template('index copy 4.html')

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

    global verificador_de_riego
    print('VERIFICANDO CUADRANTE')
    primer_riego = True
    global analysis_started
    global position_message
    while True:
        try:
            with urllib.request.urlopen(url2) as img_response:
                img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                # Redimensionar la imagen si es necesario
                resized_img = cv2.resize(img, (640, 480))

                # Convertir la imagen a espacio de color HSV
                hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

                # Definir el rango de color amarillo en HSV
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([30, 255, 255])

                # Crear una máscara para detectar el color amarillo
                mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

                # Encontrar contornos en la máscara
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Dibujar contornos alrededor de las áreas detectadas y encontrar el centro de la pelota
                ball_center = None
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Ignorar contornos pequeños
                        x, y, w, h = cv2.boundingRect(contour)
                        cv2.rectangle(resized_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        ball_center = (x + w // 2, y + h // 2)
                        cv2.circle(resized_img, ball_center, 5, (0, 255, 255), -1)
                        break

                # Obtener las dimensiones de la imagen
                height, width, _ = resized_img.shape

                # Calcular las posiciones de las líneas
                third_width = width // 3

                # Dibujar las líneas verticales
                color = (0, 255, 0)  # Color verde en BGR
                thickness = 2  # Grosor de la línea

                cv2.line(resized_img, (third_width, 0), (third_width, height), color, thickness)
                cv2.line(resized_img, (2 * third_width, 0), (2 * third_width, height), color, thickness)

                # Determinar en qué parte de la imagen está la pelota
                if ball_center is not None:
                    if ball_center[0] < third_width:
                        position_message = 'Pelota detectada a la izquierda'
                        if verificador_de_riego[2] == "no regado" and verificador_de_riego[1] == "regado" and verificador_de_riego[0] == "regado":
                            analysis_started = True
                            _, img_encoded = cv2.imencode('.jpg', resized_img)
                            img_base6 = base64.b64encode(img_encoded).decode('utf-8')
                            socketio.emit('video_posicion', {'img': img_base6, 'position_message':position_message})
                            #socketio.emit('position_message', position_message)
                            print('Análisis de video iniciado')
                            print(position_message)
                            primer_riego = False
                            threading.Thread(target=obtener_video).start()
                            break

                    elif ball_center[0] < 2 * third_width:
                        position_message = 'Pelota detectada en el centro'
                        if verificador_de_riego[1] == "no regado" and verificador_de_riego[0] == "regado":
                            analysis_started = True
                            _, img_encoded = cv2.imencode('.jpg', resized_img)
                            img_base6 = base64.b64encode(img_encoded).decode('utf-8')
                            socketio.emit('video_posicion', {'img': img_base6, 'position_message':position_message})
                            #socketio.emit('position_message', position_message)
                            print('Análisis de video iniciado')
                            print(position_message)
                            primer_riego = False
                            threading.Thread(target=obtener_video).start()
                            break

                    else:
                        position_message = 'Pelota detectada a la derecha'
                        if verificador_de_riego[0]== "no regado":
                            analysis_started = True
                            _, img_encoded = cv2.imencode('.jpg', resized_img)
                            img_base6 = base64.b64encode(img_encoded).decode('utf-8')
                            socketio.emit('video_posicion', {'img': img_base6, 'position_message':position_message})
                            #socketio.emit('position_message', position_message)
                            print('Análisis de video iniciado')
                            print(position_message)
                            primer_riego = False

                            threading.Thread(target=obtener_video).start()
                            break
                    #print(position_message)
                    #socketio.emit('position_message', position_message)
                else:
                    position_message = 'NOSE DECTECTO LA CAMARA'
                    #time.sleep(3.1)
                print(position_message)
                print(verificador_de_riego)
                _, img_encoded = cv2.imencode('.jpg', resized_img)
                img_base6 = base64.b64encode(img_encoded).decode('utf-8')

                socketio.emit('video_posicion', {'img': img_base6, 'position_message':position_message})
                time.sleep(0.1)  # Controla la velocidad de actualización del video

        except Exception as e:
            print(f"Error al obtener la imagen: {e}")
            time.sleep(1)  # Esperar antes de intentar nuevamente en caso de error

if __name__ == '__main__':
    socketio.run(app, debug=True)

