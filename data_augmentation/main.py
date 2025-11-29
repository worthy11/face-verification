import os
import cv2
import random
import gc
import torch
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
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    makeup_folder = os.path.join(script_dir, 'examples', 'makeup_photos')
    clown_path = os.path.join(script_dir, 'examples', 'clown.jpg')

    
    try:
        makeup_files = [
            os.path.join(makeup_folder, f)
            for f in os.listdir(makeup_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
    except Exception:
        makeup_files = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    person_folders = [
        d for d in sorted(os.listdir(DATASET_DIR))
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ][:5]

    for person_name in tqdm(person_folders, desc="Przetwarzanie osób"):
        person_dir = os.path.join(DATASET_DIR, person_name)

        images = sorted([
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])[:]

        output_person_dir = os.path.join(OUTPUT_DIR, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        processed_count = 0

        for img_path in images:
            img_name = os.path.basename(img_path)
            source = cv2.imread(img_path)
            if source is None:
                continue

            try:
                h, w = source.shape[:2]
            except Exception:
                continue
            if h < 200 or w < 200:
                continue
      
            if makeup_files:
                chosen = random.choice(makeup_files)
                joker = cv2.imread(chosen)
                if joker is None:
                    
                    joker = cv2.imread(clown_path)
            else:
                joker = cv2.imread(clown_path)

            result = get_makeup_transfer_results256(source, joker)
            if result is None:
                continue
            timestamp = int(time.time() * 1000)  
            name, ext = os.path.splitext(img_name)
            new_name = f"{name}_{timestamp}{ext}"
            output_path = os.path.join(output_person_dir, new_name)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            saved = cv2.imwrite(output_path, result_bgr)
            if saved:
                processed_count += 1
            
            try:
                del source, joker, result, result_bgr, chosen
            except NameError:
                try:
                    del source, joker, result, result_bgr
                except NameError:
                    pass
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            if processed_count >= 50:
                try:
                    tqdm.write(f"Przetworzono {processed_count} zdjęć dla {person_name}, przechodzę dalej.")
                except Exception:
                    pass
                break

if __name__ == '__main__':
    main()
