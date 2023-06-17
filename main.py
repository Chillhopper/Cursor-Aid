import cv2
import numpy as np
import pyautogui

cam = cv2.VideoCapture(0)
startF = (0,0)
endF = (0,0)
startE1 = (0,0)
endE1 = (0,0)
startE2 = (0,0)
endE2 = (0,0)


def prtTup(tuple):
    for num in tuple:
        print(num)

def main():
    
    check, frame = cam.read()
    #cv2.imshow('vedio', frame)
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(grey_frame, 1.3, 4)
    #print("no. of faces: %s" % len(faces))
    roi_grey = None
    for (x, y, w, h) in faces:
        global startF, endF
        startF = (x, y)
        endF = (x+w, y+h)
        roi_grey = grey_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]


    eyes = eye_cascade.detectMultiScale(grey_frame, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    print("type is %s" % type(eyes))
    print("no. of eyes %s" % len(eyes))
    if(len(eyes) > 2):
        eyes = eyes[:2]
    for eyeobj in eyes:
        global startE1, endE1
        x2 = eyeobj[0]
        y2 = eyeobj[1]
        w2 = eyeobj[2]
        h2 = eyeobj[3]
        startE1 = (x2,y2)
        endE1 = (x2+w2, y2+h2)

        color_eye = (255, 0, 0)
        cv2.rectangle(frame, startE1, endE1, color_eye, 2)    
    # for (x2, y2, w2, h2) in eyes:
    #     global startE, endE
    #     startE1 = (x2, y2)
    #     endE1 = (x2+w2, y2+h2)

    color_face = (0, 255, 0)
    color_eye = (255, 0, 0)
    cv2.rectangle(frame, startF, endF, color_face, 2)
    cv2.rectangle(frame, startE1, endE1, color_eye, 2)    
    cv2.imshow('myeProject', frame)
        

while True:

    main()
    key = cv2.waitKey(1)
    if key == 27:
        cam.release()
        cv2.destroyAllWindows()
        break