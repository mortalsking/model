from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import numpy as np
import pickle
import sqlite3
from datetime import datetime

app = Flask(__name__)

# ===== Setup =====
data_dir = "face_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

face_data = []
face_labels = []
label_names = {}
label_counter = 0

# Load saved model and data if exists
model_path = os.path.join(data_dir, "face_model.xml")
label_path = os.path.join(data_dir, "label_data.pkl")
training_path = os.path.join(data_dir, "training_data.pkl")

if os.path.exists(model_path):
    recognizer.read(model_path)
    with open(label_path, "rb") as f:
        temp = pickle.load(f)
        label_names = temp["label_names"]
        label_counter = temp["label_counter"]
    with open(training_path, "rb") as f:
        temp = pickle.load(f)
        face_data = temp["face_data"]
        face_labels = temp["face_labels"]

# SQLite Setup
conn = sqlite3.connect('attendance.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (name TEXT, timestamp TEXT)''')
conn.commit()

# ===== Utilities =====
def save_model():
    recognizer.write(model_path)
    with open(label_path, "wb") as f:
        pickle.dump({"label_names": label_names, "label_counter": label_counter}, f)
    with open(training_path, "wb") as f:
        pickle.dump({"face_data": face_data, "face_labels": face_labels}, f)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray, face_cascade.detectMultiScale(gray, 1.3, 5)

def train_model():
    if len(face_data) > 0:
        recognizer.train(face_data, np.array(face_labels))
        save_model()

# ===== Routes =====
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return redirect(url_for('video_feed_register', name=request.form['name']))
    return render_template('register.html')

@app.route('/register_feed/<name>')
def video_feed_register(name):
    def gen():
        global label_counter
        cap = cv2.VideoCapture(0)
        samples = 0
        new_label = None

        # Check if name already exists
        if name in label_names.values():
            for label, existing_name in label_names.items():
                if existing_name == name:
                    new_label = label
                    break
        else:
            new_label = label_counter
            label_names[new_label] = name
            label_counter += 1

        while samples < 20:
            ret, frame = cap.read()
            if not ret:
                break
            gray, faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (100, 100))
                face_data.append(face_img)
                face_labels.append(new_label)
                samples += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
        train_model()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognize')
def recognize():
    return render_template('recognize.html')

@app.route('/recognize_feed')
def recognize_feed():
    def gen():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray, faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                try:
                    label, confidence = recognizer.predict(face_roi)
                    name = label_names.get(label, "Unknown") if confidence < 70 else "Unknown"
                except:
                    name = "Unknown"
                if name != "Unknown":
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute("INSERT INTO attendance (name, timestamp) VALUES (?, ?)", (name, timestamp))
                    conn.commit()
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    cursor.execute("SELECT * FROM attendance ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    return render_template('attendance.html', records=rows)

if __name__ == '__main__':
    app.run(debug=True)
