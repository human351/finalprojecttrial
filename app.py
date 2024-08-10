import cv2
import base64
import numpy as np
from ultralytics import YOLO, solutions
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

model = YOLO('yolov8n.pt')
classes_to_count = [0] # 0 for person 

counter = solutions.ObjectCounter(view_img=True, reg_pts=[(20, 400), (1080, 400)], names=model.names, draw_tracks=True, line_thickness=2 )
def encode_frame(frame):
    """Encode the frame as base64 for JSON serialization."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def process_video():
    """Process video feed and emit frames to client."""
    cap = cv2.VideoCapture(0)  # Use 0 for webcam feed
    if not cap.isOpened():
        print("Error: Could not open video feed")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame")
            break
        
        # Process the frame
        processed_frame = model.track(frame, persist=True, show=False, classes=classes_to_count)
        processed_frame = counter.start_counting(frame, processed_frame)
        
        # Encode frames for transmission
        original_encoded = encode_frame(frame)
        processed_encoded = encode_frame(processed_frame)
        
        # Emit frames to the client
        socketio.emit('video_frame', {'original': original_encoded, 'processed': processed_encoded})
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(target=process_video)

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')
