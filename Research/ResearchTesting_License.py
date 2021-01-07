# Import Library
import cv2
from openalpr import Alpr
import os
import pandas as pd
from tqdm import tqdm
import pprint
import imutils
import difflib
import Levenshtein
import datetime
import sys

alpr_config_country = "id"
alpr_config_confFile = "../parameter/OpenALPR/openalpr.default.conf"
alpr_config_runtime = "../parameter/OpenALPR/runtime_data"
alpr_config_prewarp = "../planar, 759.000000,1024.000000,0.000050,0.000100,0.090000,1.000000,1.025000,0.000000,0.000000"

# Load the ALPR Module
alpr = Alpr(alpr_config_country, alpr_config_confFile, alpr_config_runtime)

if not alpr.is_loaded():
    print("Error loading OpenALPR")
    sys.exit(1)

# Set the Parameter for ALPR
alpr.set_top_n(10)
alpr.set_default_region("id")

# Prewarp Settings
# alpr.set_prewarp(prewarp)

# Testing Path
Testing_Path = "../dataset/Testing/"
Output_Base = "../output/Testing_License_Normal/"

# Get the testing images path
imagePaths = os.listdir(Testing_Path)

# Counter
truePrediction = 0
totalPrediction = 0
plateFound = 0
totalPlate = 0

total_SequenceMatcher_metric = 0
total_Levenshtein_Metric = 0

df = pd.DataFrame(columns = ['Filename', 'Ground Truth',
                            'Predicted', 'True/False',
                            'Sequence Matcher', 'Levenshtein Distance', 'Class'
                            ])

time_start = datetime.datetime.now()
print("[INFO] Start Testing")
# Start testing the dataset
# Loop through each images in the testing directory
for person in tqdm(imagePaths):
    # Get the image list of the person
    imageList = os.listdir(Testing_Path + person)

    # Loop through each images
    for person_img in imageList:
        # Get the license plate from string
        groundTruth = person_img.split('_')[1]

        print(person_img)

        # Output List
        list_output = []

        # Append the output list
        list_output.append(str(person_img))
        list_output.append(str(groundTruth))

        # Load the Input
        image_path = Testing_Path +  person + "/" + person_img
        plate_image = cv2.imread(image_path)

        # # Select the region of the license plate only
        # height = plate_image.shape[0]/2
        # width = plate_image.shape[1]/2

        # mid_to_bot_value = 2500
        # mid_to_top_value = 100

        # # x = int(0)
        # # y = int(height-mid_to_top_value)
        # # h = int(width*2)
        # # w = int(height+mid_to_bot_value)
        
        # # # plate_image = plate_image[height-y_value:height+y_value, width-x_value:width+x_value]
        # # cropped_image = plate_image[y:w, 0:x+h]

        # Resize image to reduce workload
        resized = imutils.resize(plate_image, width=512)

        # # Convert to grayscale
        # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # # Otsu tresholding
        # blur = cv2.GaussianBlur(gray,(5,5),0)
        # ret3, resized = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # # Invert the image
        # inverted = resized
        inverted = cv2.bitwise_not(resized)

        splitted = person_img.split("_")
        class_image = splitted[2] + splitted[3].split(".")[0]
        angle = splitted[3].split(".")[0]

        # # Set prewarp
        # if(angle=="20"):
        #     alpr.set_prewarp(prewarp_20)
        # elif(angle=="40"):
        #     alpr.set_prewarp(prewarp_20)
        # else:
        #     alpr.set_prewarp(prewarp_def)

        # cv2.imwrite("Test.jpg", inverted)

        # Plate Recognition
        LPR_Result = alpr.recognize_ndarray(inverted)

        # If not detected by default the result is Not Detected
        return_plate = ""
        TF = 0

        print(class_image)

        # Check the Result is Empty or Not
        if(LPR_Result['results'] != []):

            # Print all result
            # pprint.pprint(LPR_Result)
            
            Plate_Result = LPR_Result['results'][0]

            return_plate = Plate_Result['plate']
            return_confidence = Plate_Result['confidence']
            return_coordinate = Plate_Result['coordinates']

            # Convert coordinates into bounding box
            top_left = return_coordinate[0]
            top_right = return_coordinate[1]
            bot_right = return_coordinate[2]
            bot_left = return_coordinate[3]

            x1=min(top_left['x'], top_right['x'], bot_right['x'], bot_left['x'])
            y1=min(top_left['y'], top_right['y'], bot_right['y'], bot_left['y'])
            x2=max(top_left['x'], top_right['x'], bot_right['x'], bot_left['x'])
            y2=max(top_left['y'], top_right['y'], bot_right['y'], bot_left['y'])

            cv2.putText(resized, str(return_plate), (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

            cv2.rectangle(resized, (x1,y1), (x2,y2), (0, 255, 0), 2)

            # Print the Result
            print("License Plate Found")
            print("  %s %s %s %.2f" % ("ALPR Result :", return_plate, "| Confidence :", return_confidence))
            plateFound = plateFound + 1

        # Check if the license is detected
        if(return_plate != ""):
            if(return_plate == groundTruth):
                # Increase true counter
                truePrediction = truePrediction + 1
                TF = 1
        
        # Append the output list
        list_output.append(str(return_plate))
        list_output.append(str(TF))

        # Show the frame
        cv2.imshow('LP Recognition', resized)
        cv2.waitKey(1)
        
        # Calculate the metric
        SequenceMatcher_metric = difflib.SequenceMatcher(None, return_plate, groundTruth).ratio()
        Levenshtein_Metric = Levenshtein.distance(return_plate, groundTruth)
        
        total_SequenceMatcher_metric = total_SequenceMatcher_metric + SequenceMatcher_metric
        total_Levenshtein_Metric = total_Levenshtein_Metric + Levenshtein_Metric

        list_output.append(str(SequenceMatcher_metric))
        list_output.append(str(Levenshtein_Metric))

        list_output.append(str(class_image))

        # Insert output to dataframe
        # print("List value :", list_output)
        new_row = pd.Series(list_output, index = df.columns)
        df = df.append(new_row, ignore_index=True)

        # Write the Output
        Output_Path = Output_Base + person + "/" + person_img
        cv2.imwrite(Output_Path, resized)
        print(Output_Path)

        # Increase prediction counter
        totalPrediction = totalPrediction + 1

# Calculate the Training Time
time_end = datetime.datetime.now()
inference_time = time_end-time_start
print("[INFO] Inference Time : " + str(inference_time))

print("[INFO] Overall Accuracy : " + str(truePrediction/totalPrediction))
print("[INFO] Plate Found : " + str(plateFound))
print("[INFO] SequenceMatcher Metric : " + str(total_SequenceMatcher_metric/totalPrediction))
print("[INFO] Levenshtein Metric : " + str(total_Levenshtein_Metric/totalPrediction))

# Export the result
df.to_excel(Output_Base + "Report.xlsx", index = False, header=True)
print(Output_Base + "Report.xlsx")

# Shut Down the ALPR
cv2.destroyAllWindows()
alpr.unload()