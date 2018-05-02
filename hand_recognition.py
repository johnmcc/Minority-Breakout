import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

face_cascade = cv2.CascadeClassifier("./.env/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml")

while(1):
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)

    if(len(faces) > 0):
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("frame", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()