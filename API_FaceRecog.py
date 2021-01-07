# Import Library
import pycuda.autoinit
import imutils
from imutils.video import VideoStream
from imutils import paths
import pickle
import cv2
import face_recognition
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import time
import os
from sklearn import svm

# Encodings Path
Classifier_Path = "parameter/face_encodings.sav"

# Face Dataset Path
Face_Dataset_Path = "dataset/Faces/"

def fr_encode():
    # Grab the path of each face images in dataset
    imagePaths = os.listdir(Face_Dataset_Path)

    # Initialize the known encodings and face names
    encodings_list = []
    names_list = []

    print("[INFO] Start Encodings Faces!")

    encoded_counter = 0

    # Loop through each person in the training directory
    for person in tqdm(imagePaths):
        # Get the image list of the person
        imageList = os.listdir(Face_Dataset_Path + person)

        # Loop through each images
        for person_img in imageList:
            # Calculate the face encodings of each image
            face_path = Face_Dataset_Path +  person + "/" + person_img
            face = face_recognition.load_image_file(face_path)

            # Resize image to reduce workload
            resized = imutils.resize(face, width=1024)

            # Convert the Image to RGB Color Space
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Detect face in the images
            face_bounding_boxes = face_recognition.face_locations(rgb,
                                                                number_of_times_to_upsample=1,
                                                                model='cnn')

            #Verify there is one faces for each image
            if len(face_bounding_boxes) == 1:
                # Calculate the face encodings
                face_enc = face_recognition.face_encodings(rgb,
                                                        face_bounding_boxes,
                                                        num_jitters=100,
                                                        model='large')[0]

                # Append the face encoding and the face name into a list
                encodings_list.append(face_enc)
                names_list.append(person)

                encoded_counter = encoded_counter + 1

            # Extract Faces
            for top, right, bottom, left in face_bounding_boxes:
                face_crop = rgb[top:bottom, left:right]

            # Show the faces
            # cv2.imshow("Face Encoded", face_crop)
            # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    print("[INFO] Encoded Faces : " + str(encoded_counter))
    print("[INFO] Training Classifier!")
    # Create and train the SVC Classifier
    face_svm = svm.SVC(gamma='scale')
    face_svm.fit(encodings_list, names_list)
    print("[INFO] Classifier Created!")

    # Save the SVM Classifier
    pickle.dump(face_svm, open(Classifier_Path, 'wb'))

    return_string = "Encoding Succes | Encoding " + str(encoded_counter) + " Faces from " + str(len(imagePaths)) + " Class"

    return return_string

def fr_recognition(frame):
    # Load the SVM Classifier
    # print("[INFO] Load SVM Face Classifier")
    face_svmc = pickle.load(open(Classifier_Path, 'rb'))

    # Resize image to reduce workload
    resized = imutils.resize(frame, width=640)

    # Detect face in the images
    face_bounding_boxes = face_recognition.face_locations(resized,
                                                        number_of_times_to_upsample=1,
                                                        model='cnn')

    # Calculate the Face Encodings
    face_encoding_list = face_recognition.face_encodings(resized,
                                                face_bounding_boxes,
                                                num_jitters=100,
                                                model='large')

    names = []
    
    name = "Unknown"

    # Loop for each face in the image
    for face_encoding in face_encoding_list:            
        # Predict the name of the face
        name = face_svmc.predict(face_encoding.reshape(1, -1))
        
        # update the list of names
        names.append(name)

    # Loop over the detected faces
    for ((top, right, bottom, left), name) in zip(face_bounding_boxes, names):
        # Draw the rectangle on the frame
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.rectangle(resized, (left, top), (right, bottom),
            (0, 255, 0), 2)
        cv2.putText(resized, str(name), (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2)
    
    # Show the Result
    # cv2.imshow('Face Recognition Result', resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return(name, resized)

# Testing Single Frame RTSP
# RTSP_URL_KyraCAM1 = "rtsp://altius:fortius@192.168.13.41/play1.sdp"
# KyraCAM1 = VideoStream(RTSP_URL_KyraCAM1).start()
# frame = KyraCAM1.read()

# Testing Single Frame File
# frame = cv2.imread("dataset/Testing/Adhie/Adhie_B3689KDV_Distance_150.jpg")

# Start the Recognition
# fr_recognition(frame)