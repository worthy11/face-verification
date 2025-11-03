import cv2
import mediapipe as mp
import time


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

image = cv2.imread('joker.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

start = time.time()
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(image_rgb)

if results.detections:
    for detection in results.detections:
        mp_drawing.draw_detection(image, detection)

end = time.time()
print(f"Detection time: {end - start:.4f} seconds")

cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
