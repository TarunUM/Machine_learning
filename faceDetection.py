import cv2

cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier("../datasets/haarcascade_frontalface_default.xml")

while True:
    retval, frame = cap.read()

    if retval:
        faces = classifier.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face
            out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 10)
        cv2.imshow("My Window ", frame)

    key = cv2.waitKey(1)
    
    if key == ord("q"):
        break

cap.realease()
cap.destoryAllWindows()