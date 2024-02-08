import cv2
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import cv2
from datetime import datetime

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("archive/haarcascade_frontalface_alt.xml")

count = 0
skip = 0
face_data = []
dataset_path = './data/'
offset = 10



# Path to the dataset
dataset_path = './data/'

# Loading the dataset
face_data = []
labels = []

# For each file in the dataset folder, load the face data and assign labels
for filename in os.listdir(dataset_path):
    if filename.endswith('.npy'):
        # Load faces
        data = np.load(dataset_path + filename)
        face_data.append(data)
        
        # Create labels for the loaded face data
        # Assuming filename format is "PersonName.npy"
        name = filename[:-4]
        labels.append([name] * data.shape[0])

# Converting lists into numpy arrays
face_data = np.concatenate(face_data, axis=0)
labels = np.concatenate(labels, axis=0)

# Encoding labels into integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train a simple model (e.g., KNN) for face recognition
model = KNeighborsClassifier(n_neighbors=5)
model.fit(face_data, labels_encoded)

# print("Model trained successfully!")



import cv2

# Initialize the camera and load the face cascade
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("archive/haarcascade_frontalface_alt.xml")

# Initialize a list to store attendance records
attendance_records = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract the face region
        face_section = frame[y:y+h, x:x+w]
        face_section = cv2.resize(face_section, (100, 100)).flatten()

        # Predict the label of the face
        label_index = model.predict([face_section])
        label_name = label_encoder.inverse_transform(label_index)
        
        # Display the name of the person
        cv2.putText(frame, label_name[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Append the name and timestamp to the attendance records list

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        attendance_records.append([label_name[0], datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        break

cap.release()
cv2.destroyAllWindows()

# After breaking out of the loop, write the attendance records to an Excel file
if attendance_records:
    df = pd.DataFrame(attendance_records, columns=['Name', 'Timestamp'])
    df.to_excel('attendance.xlsx', index=False)
    print("Attendance sheet created successfully!")