# Import Library
import cv2
import imutils
from imutils.video import VideoStream
import mysql.connector
import datetime
import time
import random

# Import API
from API_FaceRecog import fr_recognition, fr_encode
from API_LicenseRecog import lpr_recognition
from API_ObjectRecog import object_recognition

# IP Camera RTSP URL
RTSP_URL_KyraCAM1 = "rtsp://altius:fortius@192.168.13.41/play1.sdp"
RTSP_URL_KyraCAM2 = "rtsp://altius:fortius@192.168.13.42/play1.sdp"

KyraCAM1 = VideoStream(RTSP_URL_KyraCAM1).start()
KyraCAM2 = VideoStream(RTSP_URL_KyraCAM2).start()

# Database Connection
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="parking_db"
)

mycursor = mydb.cursor()

# Base Query
Insert_Query = "INSERT INTO records (name, license_plate, timestamp, gate) VALUES (%s, %s, %s, %s)"

def insert_Records(name, license, gate):
    # Get current time
    time_now = datetime.datetime.now()
    time_now = time_now.strftime('%Y-%m-%d %H:%M:%S')
    value = (name, license, time_now, 1)
    mycursor.execute(Insert_Query, value)
    mydb.commit()
    print(mycursor.rowcount, " record inserted!")

# Forever Main Loop
while True:
    # Retrieve the Frame for OR
    frame_OR = KyraCAM1.read()

    # Process the frame
    # object_Status = object_recognition(frame_OR)
    object_Status = True

    # Process Frame
    if(object_Status == True):
        print("[INFO] Object Detected!")
        # Wait the Object to Stabilize
        time.sleep(5)

        # Get the Frame
        frame_FR = KyraCAM1.read()
        frame_LPR = KyraCAM2.read()
        
        # Detect Face
        name, fr_frame = fr_recognition(frame_FR)
        print("[INFO] Face Detected    : " + str(name[0]))

        # License Plate Detected
        license, lpr_frame = lpr_recognition(frame_LPR)
        print("[INFO] License Detected : " + str(license))

        # Insert the Data to Database
        insert_Records(name[0], license, 1)