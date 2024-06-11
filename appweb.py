from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
import threading
import time
import urllib.request
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

url = 'http://192.168.137.110/cam-hi.jpg'

"""
def obtener_video():
    while True:
        img_response = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded)
        socketio.emit('video_feed', img_base64.decode('utf-8'))
        time.sleep(0.1)  # Controla la velocidad de actualización del video
"""
def obtener_video():
    while True:
        try:
            with urllib.request.urlopen(url) as img_response:
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
                    elif ball_center[0] < 2 * third_width:
                        position_message = 'Pelota detectada en el centro'
                    else:
                        position_message = 'Pelota detectada a la derecha'
                    print(position_message)
                    socketio.emit('position_message', position_message)

                _, img_encoded = cv2.imencode('.jpg', resized_img)
                img_base64 = base64.b64encode(img_encoded).decode('utf-8')

                socketio.emit('video_feed', {'img': img_base64})
                time.sleep(0.1)  # Controla la velocidad de actualización del video

        except Exception as e:
            print(f"Error al obtener la imagen: {e}")
            time.sleep(1)  # Esperar antes de intentar nuevamente en caso de error


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    threading.Thread(target=obtener_video).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)
