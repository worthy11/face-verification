from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

nb_class = 100
hidden_dim = 512
img_size = (224, 224)
batch_size = 32
epochs = 10

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data/val',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, out)

custom_vgg_model.compile(optimizer=Adam(learning_rate=1e-4),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])


history = custom_vgg_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=1
)

custom_vgg_model.save('models/finetuned_vggface2.h5')