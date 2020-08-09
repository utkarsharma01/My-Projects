import cv2
from random import randrange

#load some pre-trained data on face frontals from opencv (hear cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
#img = cv2.imread('g.jpg')
webcam = cv2.VideoCapture(0)

#iterate forever over frames
while True:

    ###Read the current frames
    sucessful_frame_read, frame = webcam.read()

    #Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectanles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x ,y),(x+w, y+h), (randrange(256),randrange(256), randrange(256)), 2)

    cv2.imshow('Simple Programmer Face Detector', frame)
    key = cv2.waitKey(1)

    #Stop of Q key is pressed
    if key==81 or key==113:
        break
#Release the webcam
webcam.release()

print('Code Complete')
