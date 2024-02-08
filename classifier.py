import pandas as pd
import numpy as np
import cv2
import sys


# Checking if a name argument is passed
if len(sys.argv) > 1:
    file_name = sys.argv[1]  # Take the first argument as the file name
else:
    print("Error: No name provided.")
    sys.exit(1)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("archive/haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'
offset = 10

x, y, w, h = 0, 0, 0, 0  

while True:
    ret, frame = cap.read()
    
    if ret is False:
        continue
        
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2] * f[3])
    
    for face in faces[-1:]:
        x, y, w, h = face 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # extract main face
    face_section = frame[y - offset: y + h + offset, x - offset: x + w + offset]
    
    # check if face_section is not empty before resizing
    if not face_section.size == 0:
        face_section = cv2.resize(face_section, (100, 100))
        
        skip += 1
        
        if skip % 10 == 0:
            face_data.append(face_section)
            print(len(face_data))
        
        cv2.imshow("Cropped", face_section) 

    cv2.imshow("VIDEO FRAME", frame)
        
    keypressed = cv2.waitKey(5) & 0xFF
    if keypressed == ord('q'):
        break

face_data = np.array(face_data)
print(face_data.shape)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(dataset_path + file_name + '.npy', face_data)
print("Data successfully saved at " + dataset_path + file_name + '.npy')
