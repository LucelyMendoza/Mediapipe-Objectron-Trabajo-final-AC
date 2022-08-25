import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

capture=cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

with mp_objectron.Objectron(
     static_image_mode=True,
     max_num_objects=5,
     min_detection_confidence=0.5,
     model_name="Camera") as objectron: 

     while cv2.waitKey(33)<0:
        ret, image=capture.read()
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results=objectron.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detected_objects:
            for detected_object in results.detected_objects:
                 mp_drawing.draw_landmarks(image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=6),
                    mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2))
                 mp_drawing.draw_axis(image, detected_object.rotation, detected_object.translation)

        cv2.imshow("VideoFrame", image)
        key=cv2.waitKey(1)
        if key==ord("q"):
            break
capture.release()
cv2.destroyAllWindows()