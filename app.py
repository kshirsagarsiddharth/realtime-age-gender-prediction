import csv
from flask import Flask, render_template, Response
import cv2
from utils import predict_age, predict_gender

app = Flask(__name__)
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def generate_frames():

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if type(faces) == tuple:
                cv2.putText(frame, "There is no face", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                for (x, y, w, h) in faces:
                    # cv2.rectangle(frame, (x - 10, y - 60), (x + w + 10, y + h + 50),
                    #               color=(255, 0, 0), thickness=1)
                    roi_color = frame[y - 60:y+h + 50, x - 10:x+w + 10]
                    temp = frame.copy()
                    frame = cv2.blur(frame, (35,35))
                    frame[y - 60:y+h + 50, x - 10:x+w + 10] = temp[y - 60:y+h + 50, x - 10:x+w + 10]
                    try:
                        frame_age = predict_age(roi_color)
                        frame_gender = predict_gender(roi_color)

                        cv2.putText(frame, f"Age:{frame_age} G:{frame_gender}",
                                    (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    except Exception as e:
                        print("Error")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():

    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()

