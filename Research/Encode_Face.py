import pycuda.autoinit
import imutils
from imutils.video import VideoStream
from imutils import paths
import pickle
import cv2
import face_recognition
from tqdm import tqdm
import os
from sklearn import svm

# Encodings Path
Classifier_Path = "../parameter/face_encodings.sav"

# Face Dataset Path
Face_Dataset_Path = "../dataset/Faces/"

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
            cv2.imshow("Face Encoded", face_crop)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    
fr_encode()