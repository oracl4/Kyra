from flask import Flask, render_template, Response, jsonify
import cv2
import imutils
from imutils.video import VideoStream

# Import API
from API_FaceRecog import fr_recognition, fr_encode
from API_LicenseRecog import lpr_recognition

app = Flask(__name__)

# IP Camera RTSP URL
RTSP_URL_KyraCAM1 = "rtsp://altius:fortius@192.168.13.41/play2.sdp"
RTSP_URL_KyraCAM2 = "rtsp://altius:fortius@192.168.13.42/play2.sdp"

KyraCAM1 = VideoStream(RTSP_URL_KyraCAM1).start()
KyraCAM2 = VideoStream(RTSP_URL_KyraCAM2).start()

# Generate frames from video stream
def Generate_Frames_CAM1():
    # Get Frame
    frame = KyraCAM1.read()
    
    # Check the Frame
    if frame is None:
        frame = cv2.imread("resource/frame_empty.jpg")

    namem, frame = fr_recognition(frame)
    
    # Encode the frame for display
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Generate frames from video stream
def Generate_Frames_CAM2():
    # Get Frame
    frame = KyraCAM2.read()

    # Check the Frame
    if frame is None:
        frame = cv2.imread("resource/frame_empty.jpg")

    # Encode the frame for display
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# API Route
# Routing Video Feed CAM1
@app.route('/feed_cam1')
def Feed_CAM1():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(Generate_Frames_CAM1(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Routing Video Feed CAM2
@app.route('/feed_cam2')
def Feed_CAM2():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(Generate_Frames_CAM2(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Encoding Faces Route
@app.route('/encodeFaces')
def encode_Faces():
    return fr_encode()

# Default Route    
@app.route('/')
def index():
    """Kyra Parking System App Main Page"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)