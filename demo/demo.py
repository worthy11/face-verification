import cv2
import numpy as np
import os
import sys
from keras.models import load_model, Model
from keras.preprocessing import image
from keras_vggface import utils
import subprocess
import shutil
import argparse
from shutil import which
from sklearn.svm import SVC
import glob
import random

def cosine(u, v):
    return 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class FaceVerificationDemo:
    def __init__(self, model_path='../verification/models/finetuned_vggface2(2).h5', data_dir='../data'):
        self.full_model = load_model(model_path)
        self.embedding_model = Model(inputs=self.full_model.input, outputs=self.full_model.get_layer('fc7').output)
        self.reference_embedding = None
        self.svm_model = None
        self.face_detector = self._load_face_detector()
        self.data_dir = data_dir
        self._negative_images_cache = None  # Cache for negative images
    
    def _is_video_file(self, filepath):
        try:
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except:
            return False
        
    def _load_face_detector(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        return ('haar', cascade)
    
    def detect_faces(self, frame):
        faces = []
        detector_type, detector = self.face_detector
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in detections:
            faces.append((x, y, x + w, y + h))
        
        return faces
    
    def get_embedding(self, face_image):
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))

        x = np.expand_dims(face_resized, axis=0)
        x = x.astype(np.float32)
        x = utils.preprocess_input(x, version=1)
        
        embedding = self.embedding_model.predict(x, verbose=0)[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def load_reference_from_image(self, image_path):
        """Load reference image and return its embedding."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        reference_embedding = self.get_embedding(img)
        return reference_embedding
    
    def get_random_images_from_data(self, n=20, exclude_folder=None):
        """
        Get n random images from random folders in data/ directory.
        
        Args:
            n: Number of images to get
            exclude_folder: Folder name to exclude (if comparing with a specific person)
        
        Returns:
            List of image file paths
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, self.data_dir)
        
        if not os.path.exists(data_path):
            return []
        
        # Get all image files from all folders
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        all_images = []
        
        for item in os.listdir(data_path):
            folder_path = os.path.join(data_path, item)
            if os.path.isdir(folder_path):
                # Skip exclude folder if specified
                if exclude_folder and item == exclude_folder:
                    continue
                
                for ext in image_extensions:
                    all_images.extend(glob.glob(os.path.join(folder_path, ext)))
                    all_images.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        # Randomly sample n images
        if len(all_images) < n:
            return all_images
        
        return random.sample(all_images, n)
    
    def train_svm_for_video(self, reference_image_path, negative_images):
        """
        Train an SVM classifier for a video.
        
        Args:
            reference_image_path: Path to reference image (positive class)
            negative_images: List of paths to negative class images
        
        Returns:
            Trained SVM model
        """
        # Get reference embedding (positive class)
        reference_img = cv2.imread(reference_image_path)
        if reference_img is None:
            raise ValueError(f"Could not load reference image: {reference_image_path}")
        
        positive_embedding = self.get_embedding(reference_img)
        
        # Add the positive embedding multiple times to balance classes better
        # Use the same embedding but add small noise variations to make it more robust
        positive_embeddings = [positive_embedding]
        np.random.seed(42)  # For reproducibility
        for _ in range(min(19, len(negative_images))):  # Match number of negatives
            # Add small random noise (1% of magnitude) to create variations
            noise = np.random.normal(0, 0.01, positive_embedding.shape)
            noisy_embedding = positive_embedding + noise
            # Renormalize
            noisy_embedding = noisy_embedding / (np.linalg.norm(noisy_embedding) + 1e-8)
            positive_embeddings.append(noisy_embedding)
        
        # Get embeddings from negative images
        negative_embeddings = []
        for neg_img_path in negative_images:
            neg_img = cv2.imread(neg_img_path)
            if neg_img is not None:
                neg_emb = self.get_embedding(neg_img)
                negative_embeddings.append(neg_emb)
        
        if len(negative_embeddings) == 0:
            raise ValueError("No valid negative images found")
        
        # Prepare training data with balanced classes
        # Positive class: label 1
        # Negative class: label 0
        X_train = positive_embeddings + negative_embeddings
        y_train = [1] * len(positive_embeddings) + [0] * len(negative_embeddings)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train SVM
        # Try different configurations - linear kernel often works better for embeddings
        # class_weight='balanced' helps with class imbalance
        # Lower C value for more regularization
        svm = SVC(kernel='linear', probability=True, random_state=42, 
                 class_weight='balanced', C=1.0)
        svm.fit(X_train, y_train)
        
        train_pred = svm.predict(X_train)
        train_proba = svm.predict_proba(X_train)
        
        return svm
    
    def compare_embeddings(self, emb1, emb2):
        similarity = 1 - cosine(emb1, emb2)
        return similarity
    
    def process_video(self, video_path, svm_model=None, reference_image_path=None):
        """
        Process a video with a trained SVM model.
        
        Args:
            video_path: Path to video file
            svm_model: Trained SVM model for classification
            reference_image_path: Path to reference image to display alongside video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Load and prepare reference image for display
        reference_display = None
        if reference_image_path and os.path.exists(reference_image_path):
            ref_img = cv2.imread(reference_image_path)
            if ref_img is not None:
                # Resize reference image to be smaller (about 1/3 of video height)
                ref_height, ref_width = ref_img.shape[:2]
                target_ref_height = height // 3
                scale = target_ref_height / ref_height
                new_ref_width = int(ref_width * scale)
                reference_display = cv2.resize(ref_img, (new_ref_width, target_ref_height))
                
                # Create a black canvas matching video height for proper alignment
                canvas = np.zeros((height, new_ref_width, 3), dtype=np.uint8)
                # Center the reference image vertically
                y_offset = (height - target_ref_height) // 2
                canvas[y_offset:y_offset + target_ref_height, :new_ref_width] = reference_display
                reference_display = canvas
                
                # Add label to reference image
                cv2.putText(reference_display, "Reference", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Set SVM model if provided
        if svm_model is not None:
            self.svm_model = svm_model
        
        frame_count = 0
        paused = False
        
        cv2.namedWindow('Face Verification Demo', cv2.WINDOW_NORMAL)
        max_width = 1280
        max_height = 720
        
        # Adjust window size calculation if we're showing reference image
        display_width = width
        if reference_display is not None:
            display_width = width + reference_display.shape[1]
        
        if display_width > max_width or height > max_height:
            scale = min(max_width / display_width, max_height / height)
            cv2.resizeWindow('Face Verification Demo', int(display_width * scale), int(height * scale))
        
        while True:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = False
                    print("\nResumed")
                continue
            
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video reached")
                break
            
            frame_count += 1
            
            faces = self.detect_faces(frame)
            middle_height = height // 2
            filtered_faces = [(x1, y1, x2, y2) for x1, y1, x2, y2 in faces if y1 < middle_height]
            
            for face_idx, (x1, y1, x2, y2) in enumerate(filtered_faces):
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    continue
                
                current_embedding = self.get_embedding(face_crop)
                
                if self.svm_model is not None:
                    # Use SVM to predict
                    probabilities = self.svm_model.predict_proba([current_embedding])[0]
                    prediction = np.argmax(probabilities)  # Class with highest probability
                    confidence = probabilities[prediction]  # Confidence of predicted class
                    
                    is_same_identity = (prediction == 1)  # 1 = positive class (same identity)
                    color = (0, 255, 0) if is_same_identity else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    label = f"{confidence:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    label = "No ref"
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Combine frame with reference image if available
            if reference_display is not None:
                display_frame = np.hstack([reference_display, frame])
            else:
                display_frame = frame
            
            cv2.imshow('Face Verification Demo', display_frame)

            wait_time = max(1, int(1000 / fps)) if fps > 0 else 30
            key = cv2.waitKey(wait_time) & 0xFF
            
            if cv2.getWindowProperty('Face Verification Demo', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = True
                print("\nPaused")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_videos_from_folders(self, videos_folder_path='videos', loop=True):
        """
        Process videos from the new folder structure.
        Each subfolder contains a video file and a reference image (image.png).
        
        Args:
            videos_folder_path: Path to the videos folder containing numbered subfolders
            loop: If True, loop through all videos continuously
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        videos_path = os.path.join(script_dir, videos_folder_path)
        
        if not os.path.exists(videos_path):
            raise FileNotFoundError(f"Videos folder not found: {videos_path}")
        
        # Find all numbered subfolders
        video_folders = []
        for item in os.listdir(videos_path):
            folder_path = os.path.join(videos_path, item)
            if os.path.isdir(folder_path):
                # Check if folder contains a video file and reference image
                video_file = None
                reference_image = None
                
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(file.lower())
                        if ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov', '.flv', '.m4v']:
                            if self._is_video_file(file_path):
                                video_file = file_path
                        elif file.lower() == 'image.png':
                            reference_image = file_path
                
                if video_file and reference_image:
                    video_folders.append({
                        'folder': item,
                        'video': video_file,
                        'reference': reference_image
                    })
        
        if not video_folders:
            raise ValueError(f"No valid video folders found in {videos_path}")
        
        # Sort by folder name (numeric order)
        video_folders.sort(key=lambda x: int(x['folder']) if x['folder'].isdigit() else float('inf'))
        
        iteration = 0
        while True:
            iteration += 1
            for i, vf in enumerate(video_folders, 1):
                try:
                    self.svm_model = None
                    
                    negative_images = self.get_random_images_from_data(n=20)
                    svm_model = self.train_svm_for_video(vf['reference'], negative_images)
                    self.process_video(vf['video'], svm_model=svm_model, reference_image_path=vf['reference'])
                except KeyboardInterrupt:
                    return
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not loop:
                break

if __name__ == '__main__':
    demo = FaceVerificationDemo(model_path='../verification/models/finetuned_vggface2(2).h5', data_dir='../data')
    demo.process_videos_from_folders(videos_folder_path='videos', loop=True)

