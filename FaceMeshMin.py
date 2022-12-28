import cv2
import numpy as np
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=False , max_num_faces=1 , min_detection_confidence=0.5 , min_tracking_confidence=0.5)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceMesh.process(imgRGB)

    #To display the landmarks
    if results.multi_face_landmarks:
        for faceLandmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLandmarks, mpFaceMesh.FACEMESH_TESSELATION, drawSpec, drawSpec)
            mpDraw.draw_landmarks(img, faceLandmarks, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

            #To get the values/informations of each landmark
            for id,lm in enumerate(faceLandmarks.landmark):
                #print(lm)
                ih, iw, ic = img.shape #ih = Image Heigh; iw = Image Width; ic = Image Channel
                #to get the values in pixels:
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)

    #To calculate the FPS
    cTime = time.time() 
    fps = 1/(cTime - pTime) #cTime = 'Current Time' and pTime = 'Previous Time'
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3 )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
