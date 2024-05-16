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

url = 'http://192.168.100.68/cam-hi.jpg'

def obtener_video():
    while True:
        img_response = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded)
        socketio.emit('video_feed', img_base64.decode('utf-8'))
        time.sleep(0.1)  # Controla la velocidad de actualización del video

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    threading.Thread(target=obtener_video).start()

if __name__ == '__main__':
    socketio.run(app, debug=True)
