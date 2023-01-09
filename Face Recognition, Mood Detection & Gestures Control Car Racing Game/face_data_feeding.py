import cv2
import numpy as np

import os

name = input("Enter your name : ")

frames = []
outputs = []

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("../datasets/haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()

    if ret:
        faces = classifier.detectMultiScale(frame)
        cut = frame
        for face in faces:
            x, y, w, h = face

            cut = frame[y:y+h, x:x+w]

        fit = cv2.resize(cut, (100,100))
        gray = cv2.cvtColor(fit, cv2.COLOR_BGR2GRAY)

        cv2.imshow("My Window ", frame)
        cv2.imshow("My Face ", gray)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if key == ord("c"):
        frames.append(gray.flatten())
        outputs.append([name])

X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y,X])

f_name = "face_data.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)

print(data.shape)

cap.release()
cv2.destroyAllWindows()
