import cv2
import numpy as np
import face_recognition
import os
import serial
import time
import pickle

# Set the path to your images
path = r"C:\Users\msi-pc\OneDrive\face_recongnition_project\person"
images = []
classNames = []
personsList = os.listdir(path)

# Load images and names
for cl in personsList:
    curPerson = cv2.imread(os.path.join(path, cl))
    if curPerson is not None:
        images.append(curPerson)
        classNames.append(os.path.splitext(cl)[0].replace('_', ' ').upper())  # Replace underscores with spaces

print(classNames)

# Function to find encodings
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Check if encoding was found
            encodeList.append(encodings[0])
    return encodeList

# Load or create encodings
encoding_file = 'face_encodings.pkl'

# Check if the encodings file exists
if os.path.exists(encoding_file):
    with open(encoding_file, 'rb') as f:
        encodeListKnown = pickle.load(f)
    print('Loaded existing encodings.')
else:
    encodeListKnown = findEncodings(images)
    with open(encoding_file, 'wb') as f:
        pickle.dump(encodeListKnown, f)
    print('Encoding Complete and saved.')

# Start serial communication with Arduino
arduino = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino port
time.sleep(2)  # Wait for Arduino to reset

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)

    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex]
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Send signal to Arduino if "MOHAMED OSAMA" is recognized
            if name in ["MOHAMED OSAMA"]:
                arduino.write((name + '\n').encode())  # Send the name to Arduino
                arduino.write(b'BEEP\n')  # Send beep signal to Arduino

    cv2.imshow('Face Recognition', img)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
arduino.close()  # Close the serial connection
cv2.destroyAllWindows()