import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow import keras
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()
app = Flask(__name__)

speed_limit = 10

def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    if model is None:
        print("Model is not loaded. Skipping prediction.")
        return

    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = img_preprocess(image)
    image = np.array([image])
    
    try:
        steering_angle = float(model.predict(image, verbose=0)[0])
    except Exception as e:
        print(f"Error during prediction: {e}")
        steering_angle = 0.0  # Default or safe value

    throttle = 1.0 - speed / speed_limit
    print(f'Steering Angle: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.4f}')
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    global model
    model = None
    
    try:
        model = load_model('models/own_model.h5', compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load model with custom objects...")
        try:
            # If you have custom layers or loss functions, define them here
            custom_objects = {}  # Add your custom objects if any
            model = load_model('models/model.h5', custom_objects=custom_objects, compile=False)
            print("Model loaded successfully with custom objects.")
        except Exception as e:
            print(f"Error loading model with custom objects: {e}")
    
    if model is None:
        print("Model failed to load. Exiting.")
        exit(1)

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)