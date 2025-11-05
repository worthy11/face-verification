import os
import cv2
import matplotlib.pyplot as plt
import CSD_MT_eval
from tqdm import tqdm
import time

def get_makeup_transfer_results256(non_makeup_img, makeup_img):
    transfer_img = CSD_MT_eval.makeup_transfer256(non_makeup_img, makeup_img)
    return transfer_img

DATASET_DIR = r'D:\train'
OUTPUT_DIR = r'D:\makeup-photos'

def main():
    joker = cv2.imread('examples/joker.png')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    person_folders = [
        d for d in sorted(os.listdir(DATASET_DIR))
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ][:2]

    for person_name in tqdm(person_folders, desc="Przetwarzanie osób"):
        person_dir = os.path.join(DATASET_DIR, person_name)

        images = sorted([
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])[:10]

        output_person_dir = os.path.join(OUTPUT_DIR, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        for img_path in images:
            img_name = os.path.basename(img_path)
            source = cv2.imread(img_path)
            if source is None:
                continue
            

            result = get_makeup_transfer_results256(source, joker)
            if result is None:
                continue

            timestamp = int(time.time() * 1000)  
            name, ext = os.path.splitext(img_name)
            new_name = f"{name}_{timestamp}{ext}"
            output_path = os.path.join(output_person_dir, new_name)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)

if __name__ == '__main__':
    main()
