import cv2
import numpy as np
import pyautogui

# Load the pre-trained Haar cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

# Function to move the mouse based on face position
def move_mouse(face_coordinates, screen_width, screen_height):
    face_x, face_y, face_w, face_h = face_coordinates
    # Normalize face coordinates to screen width and height
    x = int(face_x + face_w / 2)
    y = int(face_y + face_h / 2)
    normalized_x = int((x / cap.get(3)) * screen_width)
    normalized_y = int((y / cap.get(4)) * screen_height)
    pyautogui.moveTo(normalized_x, normalized_y)

# Main function to detect face and move mouse
def main():
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        move_mouse((x, y, w, h), screen_width=1920, screen_height=1080)

    cv2.imshow('img', img)

# Main loop
while True:
    main()
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
