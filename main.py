import cv2
import mediapipe as mp
cam = cv2.VideoCapture(2)

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cam.read()

    frame.flags.writeable = False
    results = holistic_model.process(frame) 
    frame.flags.writeable = True

    mp_drawing.draw_landmarks(
        frame,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        frame,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == (27):
        cam.release()
        cv2.destroyAllWindows()
        break











