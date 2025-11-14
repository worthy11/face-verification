import cv2
import mediapipe as mp
import time
import os
import matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

data_dir = r'D:\train\n000002'

image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
image_files = image_files[:10]

detected_images = []

start = time.time()

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    for filename in image_files:
        image_path = os.path.join(data_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image_rgb, detection)

        detected_images.append(image_rgb)


end = time.time()
print(f"Total processing time: {end - start:.2f} seconds")

plt.figure(figsize=(15, 8))
for i, img in enumerate(detected_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image {i+1}")

plt.tight_layout()
plt.show()
