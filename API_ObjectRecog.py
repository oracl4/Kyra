import imutils
import cv2
import numpy as np
import time
from imutils.video import FPS, VideoStream
import datetime

# Model Path
model_path = "parameter/Model/MobileNetSSD_deploy.caffemodel"
proto_path = "parameter/Model/MobileNetSSD_deploy.prototxt"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

Confidence_Value = 0.6

CONSIDER = set(["motorbike"])

net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

def object_recognition(frame):
    
    # Resize the frame
    frame = imutils.resize(frame, width=640)

    (h, w) = frame.shape[:2]
    
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (384, 384), 127.5)

    object_detected = False
    
    # Predict the Object
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > Confidence_Value:
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] in CONSIDER:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                objCentroid = (int((startX+endX)/2), int((startY+endY)/2))
                y = startY - 15 if startY - 15 > 15 else startY + 15
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

                # Set the Value to True
                object_detected = True

    # Show the Result
    # cv2.imshow('Object Detection Result', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return object_detected


# Testing Single Frame RTSP
# RTSP_URL_KyraCAM1 = "rtsp://altius:fortius@192.168.13.41/play1.sdp"
# KyraCAM1 = VideoStream(RTSP_URL_KyraCAM1).start()
# frame = KyraCAM1.read()

# # Testing Single Frame File
# frame = cv2.imread("dataset/Testing/Mahdi/Mahdi_B3052SMD_Distance_100.jpg")

# # Start the Recognition
# object_recognition(frame)