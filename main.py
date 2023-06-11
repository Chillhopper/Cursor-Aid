import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    check, frame = cam.read()
    cv2.imshow('vedio', frame)
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    key = cv2.waitKey(1)
    if key == 27:
        cam.release()
        cv2.destroyAllWindows()
        break