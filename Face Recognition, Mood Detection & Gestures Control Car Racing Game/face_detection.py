import cv2

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

        fit = cv2.resize(cut, (200,200))

        cv2.imshow("My Window ", frame)
        cv2.imshow("My Face ", fit)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
