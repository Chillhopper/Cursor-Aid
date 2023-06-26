import cv2
import numpy as np
import pyautogui #width=1920, height=1080
import dlib

cam = cv2.VideoCapture(0)
startF = (0,0)
endF = (0,0)
startE1 = (0,0)
endE1 = (0,0)
startE2 = (0,0)
endE2 = (0,0)

leftE = [36, 37, 38, 39, 40, 41]
rightE = [42, 43, 44, 45, 46, 47]

def prtTup(tuple):
    x,y = tuple
    print(f"{x},{y}")

def mseXY(tuple):
    x,y = tuple
    pyautogui.moveTo(x, y, duration = 1)

def coordTaker(shape, dtype = "int"):
    coords = np.zeros((68,2), dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def thresholdTool(threshval):
    check, img = cam.read()
    grey_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, imgThresh = cv2.threshold(grey_frame, threshval, 255, cv2.THRESH_BINARY)
    cv2.imshow("Threshold test", imgThresh)


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
    print("no. of eyes %s" % len(eyes))
    if(len(eyes) >= 2):
        global startE1, endE1, startE2, endE2
        eyes = eyes[:2]

    for (x, y, w, h) in eyes:
        startE1 = (x, y)
        endE1 = (x + w, y + h)
        prtTup(startE1)
        color_eye = (255, 0, 0)
        cv2.rectangle(frame, startE1, endE1, color_eye, 2)
    
    detect = dlib.get_frontal_face_detector()
    frontal_lst = detect(grey_frame, 1)
    recognizer = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    for (i, frontal) in enumerate(frontal_lst):
        shape = recognizer(grey_frame, frontal)
        coords = coordTaker(shape)
        for(x,y) in coords:
            cv2.circle(frame, (x,y), 2, (0, 0, 255), -1)
    
    
    color_face = (0, 255, 0)
    color_eye = (255, 0, 0)
    cv2.rectangle(frame, startF, endF, color_face, 2)
    cv2.imshow('myeProject', frame)

        

while True:

    main()

    key = cv2.waitKey(1)
    if key == 27:
        cam.release()
        cv2.destroyAllWindows()
        break








#------------------------------The Abyss------------------------------
def locker():
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
    print("no. of eyes %s" % len(eyes))
    if(len(eyes) >= 2):
        global startE1, endE1, startE2, endE2
        eyes = eyes[:2]
        
        # eyeobj1 = eyes[1]
        # x1 = eyeobj1[0]
        # y1 = eyeobj1[1]
        # w1 = eyeobj1[2]
        # h1 = eyeobj1[3]
        # startE1 = (x1,y1)
        # endE1 = (x1+w1, y1+h1)
        # eyeobj2 = eyes[1]
        # x2 = eyeobj2[0]
        # y2 = eyeobj2[1]
        # w2 = eyeobj2[2]
        # h2 = eyeobj2[3]
        # startE2 = (x2, y2)
        # endE2 = (x2+w2, y2+h2)
        # color_eye = (255, 0, 0)
        # cv2.rectangle(frame, startE1, endE1, color_eye, 2)
        # cv2.rectangle(frame, startE2, endE2, color_eye, 2)
        
        
        for x, y, w, h in eyes:
            startE1 = (x, y)
            endE1 = (x + w, y + h)
            prtTup(startE1)
            color_eye = (255, 0, 0)
            cv2.rectangle(frame, startE1, endE1, color_eye, 2)

        # for eyeobj in eyes:
        #     global startE, endE
        #     startE1 = (eyeobj[0], eyeobj[1])
        #     endE1 = (eyeobj[0] + eyeobj[2], eyeobj[1] + eyeobj[3])
        #     color_eye = (255, 0, 0)
        #     cv2.rectangle(frame, startE1, endE1, color_eye, 2)
        #cv2.rectangle(frame, startE2, endE2, color_eye, 2)  
        #mseXY(startE1)    

         
    #prtTup(startE1)
    color_face = (0, 255, 0)
    color_eye = (255, 0, 0)
    cv2.rectangle(frame, startF, endF, color_face, 2)