import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

data = np.load("face_data.npy")

X = data[:, 1:].astype(np.uint8) #int or np.uint8
y = data[:, 0]

model = KNeighborsClassifier()

model.fit(X,y)

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

            out = model.predict([gray.flatten()])

            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, str(out[0]), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 2,(255,0,0))
            # cv2.imshow("My Face ", gray)

        cv2.imshow("My Window ", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()