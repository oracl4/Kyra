# Import Library
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os
from tqdm import tqdm
import pandas as pd

# Encodings Path
Classifier_Path = "../parameter/face_encodings.sav"

# Testing Path
Testing_Path = "../dataset/Testing/"

# Load the SVM Classifier
print("[INFO] Load SVM Face Classifier")
face_svmc = pickle.load(open(Classifier_Path, 'rb'))

# Get the testing images path
imagePaths = os.listdir(Testing_Path)

# Counter
truePrediction = 0
totalPrediction = 0

df = pd.DataFrame(columns = ['Filename', 'Ground Truth Name',
                            'Predicted Name', 'True/False', 'Class'
                            ])

print("[INFO] Start Testing")
# Start testing the dataset
# Loop through each images in the testing directory
for person in tqdm(imagePaths):
    # Get the image list of the person
    imageList = os.listdir(Testing_Path + person)
    
    name_groundTruth = person

    # Loop through each images
    for person_img in imageList:
        # Output List
        list_output = []

        # Calculate the face encodings of each image
        image_path = Testing_Path +  person + "/" + person_img
        frame = face_recognition.load_image_file(image_path)

        list_output.append(str(person_img))
        list_output.append(str(name_groundTruth))

        # Resize image to reduce workload
        resized = imutils.resize(frame, width=1024)

        # # Rotate the frame
        if(person != "Martinus"):
            resized = imutils.rotate_bound(resized, 90)

        # Convert the Image to RGB Color Space
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Detect face in the images
        face_bounding_boxes = face_recognition.face_locations(rgb,
                                                            number_of_times_to_upsample=1,
                                                            model='cnn')

        # Calculate the Face Encodings
        face_encoding_list = face_recognition.face_encodings(rgb,
                                                    face_bounding_boxes,
                                                    num_jitters=100,
                                                    model='large')

        names = []
        
        name = "Not Detected"
        TF = 0

        # Loop for each face in the image
        for face_encoding in face_encoding_list:            
            # Predict the name of the face
            name = face_svmc.predict(face_encoding.reshape(1, -1))
            
            # update the list of names
            names.append(name)

        list_output.append(str(name))

        # Check if face is detected
        if(name != "Not Detected"):
            if(name == name_groundTruth):
                # Increase true counter
                truePrediction = truePrediction + 1
                TF = 1
        
        list_output.append(str(TF))
        splitted = person_img.split("_")
        class_image = splitted[2] + splitted[3].split(".")[0]

        print(name)
        list_output.append(str(class_image))

        # Loop over the detected faces
        for ((top, right, bottom, left), name) in zip(face_bounding_boxes, names):
            # Draw the rectangle on the frame
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.rectangle(rgb, (left, top), (right, bottom),
                (0, 255, 0), 2)
            cv2.putText(rgb, str(name), (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)
        
        # Increase prediction counter
        totalPrediction = totalPrediction + 1

        # Write the Output
        Output_Path = "../output/Testing_Faces/" + person + "/" + person_img
        cv2.imwrite(Output_Path, rgb)

        # Insert output to dataframe
        # print("List value :", list_output)
        new_row = pd.Series(list_output, index = df.columns)
        df = df.append(new_row, ignore_index=True)

        # Show the frame
        cv2.imshow("Face Recognition", rgb)
        cv2.waitKey(1)

print("[INFO] Accuracy : " + str(truePrediction/totalPrediction))

# Export the result
df.to_excel("../output/Testing_Faces/Report.xlsx", index = False, header=True)

cv2.destroyAllWindows()