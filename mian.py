import cv2
from picamera2 import Picamera2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time
import RPi.GPIO as GPIO
from gpiozero import DistanceSensor
import imutils
import face_recognition

import pickle

BUTTON_PIN=17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN)

print("Press button to start")
while True:
    
    if(GPIO.input(BUTTON_PIN)):
        break
print("Program starts")

#ultrasonic sensor setup
ultrasonic =DistanceSensor(echo=24, trigger=18)


#Facial recognition initialize
# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
tolerance = 0.5  # Set your desired tolerance level here

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
with open(encodingsP, "rb") as f:
    data = pickle.load(f)

data_encodings=data["encodings"]
data_names=data["names"]


#camera initialization 
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640,480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

#custom model load
model=YOLO('best.pt')
my_file = open("coco1.txt", "r")

#pretrained model load
#model=YOLO('yolov8n.pt')
#my_file = open("coco.txt", "r")

data = my_file.read()
class_list = data.split("\n")





#wait 3 seconds to update return values(largest_object_over_time, distance_over_time)
TIME_RESOLUTION=3

#return value initialization
largest_object_over_time=None
distance_over_time=None
largest_face_over_time=None


last_update_time=time.time()
while True:
    im= picam2.capture_array()

    results=model.predict(im)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    

    largest_area = 0
    largest_object = None
    
    largest_face_area = 0
    largest_face = None
    
    object_list=[]
    
    for index,row in px.iterrows():

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        object_list.append(c)
        #calculatet the area of the bounding box
        area=(x2-x1)*(y2-y1)
        
        #Update the largest object if the current one is bigger
        if area > largest_area:
            largest_area=area
            largest_object=(x1, y1, x2, y2, c)
        
        
        cv2.rectangle(im,(x1,y1),(x2,y2),(0,0,255),2)
        cvzone.putTextRect(im,f'{c}',(x1,y1),1,1)
        
        
    if("person" in object_list):
        
        # Detect the face boxes
        boxes = face_recognition.face_locations(im)
        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(im, boxes)
        names = []
        
        
        
        # loop over the facial embeddings
        for encoding in encodings:
            # compute distances to all known faces
            distances = face_recognition.face_distance(data_encodings, encoding)
            min_distance = min(distances)  # find the closest match
            name = "Unknown"  # default name if no close match is found

            # check if the minimum distance is within the tolerance
            if min_distance < tolerance:
                # find the index of the closest match
                best_match_index = distances.argmin()
                name = data_names[best_match_index]


            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            cv2.rectangle(im, (left, top), (right, bottom), (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(im, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

            #calculatet the area of the bounding box
            face_area=(bottom-top)*(right-left)
            
            #Update the largest face if the current one is bigger
            if face_area > largest_face_area:
                largest_face_area=face_area
                largest_face=(top, right, bottom, left, name)
                print(largest_face)
                
    
    

    

    current_time=time.time()
    if current_time - last_update_time>=3:
        if largest_object:
            largest_object_over_time = largest_object
        if ultrasonic.distance:
            distance_over_time=ultrasonic.distance
        if largest_face:
            largest_face_over_time = largest_face
        last_update_time=current_time
    
    #print largest_object, largest_object_over_time, ultrasonic_detected
#     if largest_object:
#         x1, y1, x2, y2, c = largest_object
#         cvzone.putTextRect(im,f'Largest object:{c}',(0,20),1,1)
        
    if largest_object_over_time:
        x1, y1, x2, y2, c = largest_object_over_time
        cvzone.putTextRect(im,f'Object detected:{c}',(0,20),1,1)

    if distance_over_time:
        cvzone.putTextRect(im,f'Distance :{distance_over_time}',(0,50),1,1)
    
    if largest_face_over_time:
        top, right, bottom, left, name = largest_face_over_time
        cvzone.putTextRect(im,f'Face recognized :{name}',(0,80),1,1)
        
    cv2.imshow("Camera", im)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
