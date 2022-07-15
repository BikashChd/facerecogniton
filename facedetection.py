import cv2 

import numpy as np
from random import randrange
# face and eye detection

cap=cv2.VideoCapture(0)
# load some pre_trained data on face frontal from open cv(haarscascade algorithm)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# load soome pre-trained data on eye from open cv(haarscascade algorithm)
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret,frame=cap.read()
    # converting frame to greyscale
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # to detect face
    faces=face_cascade.detectMultiScale(grey,1.3,5)
    
    
    for (x,y,w,h) in faces:
        # drawing rectangle in the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),5)
        # slicing the array of pixel position of  the face  for eye detection
        roi_grey=grey[y:y+w,x:x+w]
        # slicing the array of pixel position to be displayed on the screen
        roi_color=frame[y:y+h,x:x+w]
        # to detect eye
        eyes=eye_cascade.detectMultiScale(roi_grey,1.3,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(randrange(255),randrange(256),randrange(256)))


    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()