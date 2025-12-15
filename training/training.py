import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import random
from pathlib import Path


IMG_SIZE = (224, 224)
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
MARGIN = 0.2
DATA_DIR = 'data/train'
VALIDATION_SPLIT = 0.2


def create_resnet_backbone(input_shape=(224, 224, 3), embedding_dim=128):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    inputs = Input(shape=input_shape, name='input')
    x = base_model(inputs, training=False)
    
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.3, name='dropout1')(x)
    x = layers.Dense(256, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    
    embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name='embed')(x)
    embeddings = layers.Dense(embedding_dim, name='embedding_output')(embeddings)
    embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), name='norma_embed')(embeddings)
    
    model = Model(inputs, embeddings, name='resnet_embedding_model')
    return model


class TripletLoss(keras.losses.Loss):
    def __init__(self, margin=0.2, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        batch_size = tf.shape(y_pred)[0] // 3
        
        anchor = y_pred[0:batch_size]
        positive = y_pred[batch_size:2*batch_size]
        negative = y_pred[2*batch_size:3*batch_size]
        
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)


def get_data_structure(data_dir):
    data_structure = {}
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Data directory {data_dir} does not exist!")
    
    for person_dir in data_path.iterdir():
        if person_dir.is_dir():
            person_id = person_dir.name
            image_files = [str(f) for f in person_dir.iterdir() 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            if len(image_files) > 0:
                data_structure[person_id] = image_files
    
    return data_structure


class TripletGenerator(keras.utils.Sequence):
    def __init__(self, data_structure, batch_size=32, img_size=(224, 224), 
                 shuffle=True, augment=True):
        self.data_structure = data_structure
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        
        self.image_list = []
        self.person_to_indices = {}
        
        for idx, (person_id, images) in enumerate(data_structure.items()):
            person_indices = []
            for img_path in images:
                self.image_list.append((person_id, img_path))
                person_indices.append(len(self.image_list) - 1)
            self.person_to_indices[person_id] = person_indices
        
        self.person_ids = list(data_structure.keys())
        self.indices = np.arange(len(self.image_list))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        anchors = []
        positives = []
        negatives = []
        
        for i in batch_indices:
            person_id, anchor_path = self.image_list[i]
            
            person_images = self.person_to_indices[person_id]
            positive_idx = random.choice([p for p in person_images if p != i])
            _, positive_path = self.image_list[positive_idx]
            
            negative_person = random.choice([p for p in self.person_ids if p != person_id])
            negative_path = random.choice(self.data_structure[negative_person])
            
            anchors.append(self.load_and_preprocess_image(anchor_path))
            positives.append(self.load_and_preprocess_image(positive_path))
            negatives.append(self.load_and_preprocess_image(negative_path))
        
        images = np.concatenate([anchors, positives, negatives], axis=0)
        labels = np.zeros(len(images))
        
        return images, labels
    
    def load_and_preprocess_image(self, img_path):
        img = load_img(img_path, target_size=self.img_size)
        img = img_to_array(img)
        img = img.astype('float32') / 255.0
        
        if self.augment:
            img = self.apply_augmentation(img)
        
        return img
    
    def apply_augmentation(self, img):
        if np.random.random() > 0.5:
            img = np.fliplr(img)
        
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
        
        if np.random.random() > 0.5:
            img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)
        
        return img
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_embedding_model(backbone_type='resnet', input_shape=(224, 224, 3), embedding_dim=128):
    if backbone_type == 'resnet':
        return create_resnet_backbone(input_shape, embedding_dim)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}. Choose 'resnet' or 'custom'.")


def train_model(model, train_generator, val_generator, epochs=50, learning_rate=0.0001, 
                margin=0.2, model_save_path='models/face_embedding_model.h5'):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=TripletLoss(margin=margin),
        metrics=[]  # Can add distance metrics if needed
    )
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_loss' if val_generator else 'loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if val_generator else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if val_generator else 'loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator if val_generator else None,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model


def main(backbone_type='both'):
    print("Loading data structure...")
    data_structure = get_data_structure(DATA_DIR)
    
    if len(data_structure) == 0:
        raise ValueError(f"No data found in {DATA_DIR}. Expected structure: {DATA_DIR}/person_id/image.jpg")
    
    print(f"Found {len(data_structure)} persons with images")
    
    person_ids = list(data_structure.keys())
    np.random.seed(42)
    np.random.shuffle(person_ids)
    split_idx = int(len(person_ids) * (1 - VALIDATION_SPLIT))
    train_persons = person_ids[:split_idx]
    val_persons = person_ids[split_idx:]
    
    train_data = {pid: data_structure[pid] for pid in train_persons}
    val_data = {pid: data_structure[pid] for pid in val_persons}
    
    print(f"Training on {len(train_data)} persons, validating on {len(val_data)} persons")
    
    print("Creating data generators...")
    train_gen = TripletGenerator(
        train_data,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        shuffle=True,
        augment=True
    )
    
    val_gen = None
    if len(val_data) > 0:
        val_gen = TripletGenerator(
            val_data,
            batch_size=BATCH_SIZE,
            img_size=IMG_SIZE,
            shuffle=False,
            augment=False
        )
        print(f"Validation generator created with {len(val_gen)} batches")
    
    if backbone_type in ['resnet', 'both']:
        print("\n" + "="*50)
        print("Creating ResNet-based embedding model...")
        print("="*50)
        resnet_model = create_embedding_model(
            backbone_type='resnet',
            input_shape=(*IMG_SIZE, 3),
            embedding_dim=EMBEDDING_DIM
        )
        resnet_model.summary()
        
        print("\n" + "="*50)
        print("Training ResNet model...")
        print("="*50)
        resnet_history, resnet_model = train_model(
            resnet_model,
            train_gen,
            val_gen,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            margin=MARGIN,
            model_save_path='models/resnet_face_embedding.h5'
        )
        print(f"ResNet model saved to: models/resnet_face_embedding.h5")
    
    if backbone_type in ['custom', 'both']:
        print("\n" + "="*50)
        print("Creating custom CNN embedding model (no ResNet)...")
        print("="*50)
        custom_model = create_embedding_model(
            backbone_type='custom',
            input_shape=(*IMG_SIZE, 3),
            embedding_dim=EMBEDDING_DIM
        )
        custom_model.summary()
        
        print("\n" + "="*50)
        print("Training custom CNN model...")
        print("="*50)
        custom_history, custom_model = train_model(
            custom_model,
            train_gen,
            val_gen,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            margin=MARGIN,
            model_save_path='models/custom_face_embedding.h5'
        )
        print(f"Custom model saved to: models/custom_face_embedding.h5")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()

