import cv2
import numpy as np
import pyautogui #width=1920, height=1080
import dlib

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

def move_mouse(eye_coordinates, screen_width, screen_height):
    eye_x, eye_y, eye_w, eye_h = eye_coordinates
    # Normalize eye coordinates to screen width and height
    x = int(eye_x + eye_w / 2)
    y = int(eye_y + eye_h / 2)
    normalized_x = int((x / cap.get(3)) * screen_width)
    normalized_y = int((y / cap.get(4)) * screen_height)
    pyautogui.moveTo(normalized_x, normalized_y)

def main():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            move_mouse((ex, ey, ew, eh), screen_width=1920, screen_height=1080)

    cv2.imshow('img',img)


while True:
    main()
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break