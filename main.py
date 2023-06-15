import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
cam = cv2.VideoCapture(2)

BaseOptions = mp.tasks.BaseOptions
HandLandMarker = mp.tasks.vision.HandLandmarker
HandLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandMarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: HandLandMarkerResult, output_image: mp.Image, timestamp_ms: int):
    print(f'hand land marker result: {result}')

options = HandLandMarkerOptions(
    base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_cb=print_result)                            

with HandLandMarker.create_from_options(options) as landmarker:
    while True:
        ret, frame=cam.read()

        cam_array=np.array(frame)

        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=cam_array)

        landmarker.detect_async(mp_image)

        cv2.imshow(cv2.cvtColor(mp_image, cv2.COLOR_RGB2BGR)) 
        
        if cv2.waitKey(1) == (27):
            cam.release()
            cv2.destroyAllWindows()
            break

    













