import os
import numpy as np
from keras.preprocessing import image
from keras_vggface import utils
from keras.models import load_model

model = load_model('models/finetuned_vggface2(2).h5')

def verify_functionality(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)
    preds = model.predict(x)
    print('Predicted:', np.argmax(preds))

def evaluate(dir):
    total = 0
    correct = 0

    folders = [f for f in os.listdir(dir) if os.path.isdir(os.path.join(dir, f))]

    for label, folder in enumerate(folders):
        if label != 1:
            continue
        folder_path = os.path.join(dir, folder)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)

            img = image.load_img(fpath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = utils.preprocess_input(x, version=1)
            preds = model.predict(x)
            guess = np.argmax(preds)
            if guess == label:
                correct += 1
            total += 1

            print(f"\rProcessing directory {folder} ({label+1}/13): {total}, Guessed: {guess}, Accuracy: {correct/total:.4f}", end="", flush=True)

# print("Accuracy on training data:")
# evaluate('data/train')

print("\nAccuracy on augmented data:")
evaluate('data/eval')