import cv2
import numpy as np
import mediapipe as mp
import time

class FaceMeshDetector():

    def __init__(self, static_image_mode=False , max_num_faces=1 , model_complexity = 1, min_detection_confidence=0.5 , min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode , self.max_num_faces , self.model_complexity, self.min_detection_confidence , self.min_tracking_confidence)

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.faceMesh.process(self.imgRGB)
        faces = []

        #To display the landmarks
        if self.results.multi_face_landmarks:
            for faceLandmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                    self.mpDraw.draw_landmarks(img, faceLandmarks, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                #To get the values/informations of each landmark and store at face[]
                face = []
                for id,lm in enumerate(faceLandmarks.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape #ih = Image Heigh; iw = Image Width; ic = Image Channel
                    #to get the values in pixels:
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    #To visualize the id numbers inside the video, uncomment the code bellow:
                    #cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,255,255), 1 )
                    print(id,x,y)
                    face.append([x,y])
                faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))
        #To calculate the FPS
        cTime = time.time() 
        fps = 1/(cTime - pTime) #cTime = 'Current Time' and pTime = 'Previous Time'
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3 )

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()