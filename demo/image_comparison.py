import cv2
import numpy as np
import os
from keras.models import load_model, Model
from keras_vggface import utils
import time
import glob

def cosine(u, v):
    """Compute cosine distance between two vectors."""
    return 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class ImageComparisonApp:
    def __init__(self, model_path='../verification/models/finetuned_vggface2(2).h5', similarity_threshold=0.7):
        """
        Initialize the image comparison app.
        
        Args:
            model_path: Path to the finetuned VGGFace model
            similarity_threshold: Threshold for determining same/different identity
        """
        self.full_model = load_model(model_path)
        
        # Create embedding model by extracting output from fc7 layer
        try:
            fc7_layer = self.full_model.get_layer('fc7')
            self.embedding_model = Model(inputs=self.full_model.input, outputs=fc7_layer.output)
        except:
            # Fallback: use second-to-last layer
            embedding_output = self.full_model.layers[-2].output
            self.embedding_model = Model(inputs=self.full_model.input, outputs=embedding_output)
        
        self.similarity_threshold = similarity_threshold
    
    def get_embedding(self, image):
        """Extract embedding from an image (no face detection, just resize and process)."""
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        x = np.expand_dims(img_resized, axis=0)
        x = x.astype(np.float32)
        x = utils.preprocess_input(x, version=1)
        
        embedding = self.embedding_model.predict(x, verbose=0)[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def compare_images(self, img1_path, img2_path):
        """
        Compare two images and return similarity score.
        
        Returns:
            similarity: float between -1 and 1, or None if images cannot be loaded
        """
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return None
        
        # Get embeddings directly from images (no face detection)
        emb1 = self.get_embedding(img1)
        emb2 = self.get_embedding(img2)
        
        # Compute similarity
        similarity = 1 - cosine(emb1, emb2)
        return similarity
    
    def get_image_index(self, image_path):
        """Extract the starting index (first part before underscore) from image filename."""
        basename = os.path.basename(image_path)
        index = basename.split('_')[0]
        return index
    
    def categorize_images_by_index(self, image_files):
        """
        Categorize images by index and whether they have makeup (timestamp).
        
        Returns:
            images_by_index: dict mapping index -> {'no_makeup': [paths], 'makeup': [paths]}
        """
        images_by_index = {}
        
        for img_file in image_files:
            basename = os.path.basename(img_file)
            parts = basename.split('_')
            index = parts[0]
            
            if index not in images_by_index:
                images_by_index[index] = {'no_makeup': [], 'makeup': []}
            
            # Check if filename has timestamp (numbers after second underscore)
            if len(parts) >= 3 and parts[2].replace('.jpg', '').replace('.jpeg', '').replace('.png', '').isdigit():
                images_by_index[index]['makeup'].append(img_file)
            else:
                images_by_index[index]['no_makeup'].append(img_file)
        
        return images_by_index
    
    def find_two_different_indices(self, images_by_index):
        """
        Find two different indices that have both no-makeup and makeup images.
        
        Returns:
            (index1, index2) or (None, None) if not found
        """
        valid_indices = []
        for index, images in images_by_index.items():
            if images['no_makeup'] and images['makeup']:
                valid_indices.append(index)
        
        if len(valid_indices) < 2:
            return (None, None)
        
        # Return first two valid indices
        return (valid_indices[0], valid_indices[1])
    
    def get_image_by_index_and_type(self, images_by_index, index, image_type='no_makeup'):
        """
        Get an image of specified type from specified index.
        
        Args:
            images_by_index: Dictionary of images organized by index
            index: The index to look for
            image_type: 'no_makeup' or 'makeup'
        
        Returns:
            Image path or None if not found
        """
        if index in images_by_index:
            images = images_by_index[index].get(image_type, [])
            if images:
                return images[0]  # Return first available
        return None
    
    def display_pair(self, img1_path, img2_path, similarity=None, duration=2.0):
        """
        Display two images side by side.
        
        Args:
            img1_path: Path to left image (no makeup)
            img2_path: Path to right image (makeup)
            similarity: Similarity score to display
            duration: Display duration in seconds
        """
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            return
        
        # Resize images to same height
        target_height = 400
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        scale1 = target_height / h1
        scale2 = target_height / h2
        
        new_w1 = int(w1 * scale1)
        new_w2 = int(w2 * scale2)
        
        img1_resized = cv2.resize(img1, (new_w1, target_height))
        img2_resized = cv2.resize(img2, (new_w2, target_height))
        
        combined = np.hstack([img1_resized, img2_resized])
        
        if similarity is not None:
            result = "SAME" if similarity >= self.similarity_threshold else "DIFFERENT"
            color = (0, 255, 0) if similarity >= self.similarity_threshold else (0, 0, 255)
            confidence_pct = similarity * 100
            cv2.putText(combined, f"{result} (confidence {confidence_pct:.1f}%)", 
                       (10, combined.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Display
        cv2.imshow('Image Comparison', combined)
        
        # Wait for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
        return True
    
    def run(self, data_dir='../data', display_duration=2.0):
        """
        Run the image comparison app.
        
        Args:
            data_dir: Path to data directory containing person folders
            display_duration: Duration to display each pair in seconds
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, data_dir)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Get all person folders (sorted)
        person_folders = []
        for item in os.listdir(data_path):
            folder_path = os.path.join(data_path, item)
            if os.path.isdir(folder_path):
                person_folders.append((item, folder_path))
        
        person_folders.sort(key=lambda x: x[0])  # Sort by folder name
        
        if not person_folders:
            raise ValueError(f"No person folders found in {data_path}")
        
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        
        # Loop through all folders
        for folder_name, folder_path in person_folders:
            # Get all images from folder
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            if not image_files:
                continue
            
            # Categorize images by index
            images_by_index = self.categorize_images_by_index(image_files)
            
            # Get all no-makeup images
            all_no_makeup = []
            for index, images in images_by_index.items():
                all_no_makeup.extend(images.get('no_makeup', []))
            
            # Get all makeup images
            all_makeup = []
            for index, images in images_by_index.items():
                all_makeup.extend(images.get('makeup', []))
            
            if not all_no_makeup:
                continue
            
            if not all_makeup:
                continue
            
            # Select one no-makeup image for left (use first one)
            left_img = all_no_makeup[0]
            
            # Loop through up to 10 makeup images
            makeup_images_to_process = all_makeup[:10]
            
            for right_img in makeup_images_to_process:
                # Compute similarity
                similarity = self.compare_images(left_img, right_img)
                
                if similarity is not None:
                    result = "SAME" if similarity >= self.similarity_threshold else "DIFFERENT"
                    confidence_pct = similarity * 100
                    print(f"{result} (confidence {confidence_pct:.1f}%)")
                else:
                    print("Could not compute similarity")
                
                # Display pair
                if not self.display_pair(left_img, right_img, similarity, display_duration):
                    return
        
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = ImageComparisonApp(
        model_path='../verification/models/finetuned_vggface2(2).h5',
        similarity_threshold=0.75
    )
    app.run(data_dir='../data', display_duration=1.0)

