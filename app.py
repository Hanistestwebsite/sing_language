from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('model/enhanced_SL_model.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

actions = ['Hi', 'Terima Kasih', 'Bantu', 'Nama', 'Ya', 'Tidak', 'Minta', 'Saya', 'Awak', 'Maaf', 'Apa', 'Sama-sama', 'A', 'B', 'C','No Sign']

def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()
            if results.multi_handedness[0].classification[0].label == 'Left':
                lh = hand_points
            else:
                rh = hand_points
    return np.concatenate([lh, rh])

def preprocess_frame(frame):
    keypoints = extract_keypoints(frame)
    return keypoints

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    sequence = []
    sequence_length = 30  # Number of frames per sequence

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = preprocess_frame(frame)
        sequence.append(keypoints)

        if len(sequence) >= sequence_length:
            sequence_data = np.expand_dims(np.array(sequence), axis=0)  # Add batch dimension
            prediction = model.predict(sequence_data)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Display the prediction
            cv2.putText(frame, f'Prediction: {actions[predicted_class]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            sequence.pop(0)  # Remove the oldest frame to keep the sequence length constant

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

