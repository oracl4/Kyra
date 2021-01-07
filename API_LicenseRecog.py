import cv2
from openalpr import Alpr
import pprint
import imutils
from datetime import datetime
from imutils.video import VideoStream

alpr_config_country = "id"
alpr_config_confFile = "parameter/OpenALPR/openalpr.default.conf"
alpr_config_runtime = "parameter/OpenALPR/runtime_data"
alpr_config_prewarp = "planar, 759.000000,1024.000000,0.000050,0.000100,0.090000,1.000000,1.025000,0.000000,0.000000"

def lpr_recognition(frame):
    # Load the ALPR Module
    alpr = Alpr(alpr_config_country, alpr_config_confFile, alpr_config_runtime)

    # Check the Module
    if not alpr.is_loaded():
        print("Error loading OpenALPR")
        sys.exit(1)

    # Set the Parameter for ALPR
    alpr.set_top_n(10)
    alpr.set_default_region("id")

    # Prewarp Settings
    # alpr.set_prewarp(prewarp)

    # Image Input
    plate_image = frame
    
    # Resize image to reduce workload
    resized = imutils.resize(frame, width=640)

    # Invert the image
    inverted = cv2.bitwise_not(resized)

    # Plate Recognition
    LPR_Result = alpr.recognize_ndarray(inverted)

    # If not detected by default the result is Not Detected
    return_plate = "-"

    # Check the Result is Empty or Not
    if(LPR_Result['results'] != []):

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
        # print("License Plate Found")
        # print("  %s %s %s %.2f" % ("ALPR Result :", return_plate, "| Confidence :", return_confidence))
    else:
        # print("No License Plate Found in the Image")
        return_plate = "-"
    
    # Shut Down the ALPR
    alpr.unload()

    # Show the Result
    # cv2.imshow('Plate LPR Result', resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return(return_plate, resized)

# Testing Single Frame RTSP
# RTSP_URL_KyraCAM1 = "rtsp://altius:fortius@192.168.13.41/play1.sdp"
# KyraCAM1 = VideoStream(RTSP_URL_KyraCAM1).start()
# frame = KyraCAM1.read()

# Testing Single Frame File
# frame = cv2.imread("dataset/Testing/Adhie/Adhie_B3689KDV_Distance_150.jpg")

# Start the Recognition
# lpr_recognition(frame)