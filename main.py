import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cam = cv2.VideoCapture(2)

while True:
    ret, frame = cam.read()

    cv2.imshow('frame', frame) 

    if cv2.waitKey(1) == (27):
        cam.release()
        cv2.destroyAllWindows()
        break

