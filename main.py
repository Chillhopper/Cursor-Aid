import cv2
import numpy as np

cam = cv2.VideoCapture(0)
startE = (0,0)
endE = (0,0)

def main():
    
    check, frame = cam.read()
    #cv2.imshow('vedio', frame)
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(grey_frame, 1.3, 4)
    #print("no. of faces: %s" % len(faces))

    for (x, y, w, h) in faces:
        start = (x, y)
        end = (x+w, y+h)
        color_face = (0, 255, 0)
        cv2.rectangle(grey_frame, start, end, color_face, 2)
        roi_grey = grey_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]


        eyes = eye_cascade.detectMultiScale(roi_grey, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        print("no. of eyes %s" % len(eyes))
        for (x2, y2, w2, h2) in eyes:
            global startE, endE
            startE = (x2, y2)
            endE = (x2+w, y2+h)

    color_eye = (255, 0, 0)
    cv2.rectangle(frame, startE, endE, color_eye, 2)    
    cv2.imshow('myeProject', frame)
        

while True:

    main()

    key = cv2.waitKey(1)
    if key == 27:
        cam.release()
        cv2.destroyAllWindows()
        break