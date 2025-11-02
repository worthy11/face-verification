import os
import cv2
import matplotlib.pyplot as plt
import CSD_MT_eval

def get_makeup_transfer_results256(non_makeup_img,makeup_img):
    transfer_img=CSD_MT_eval.makeup_transfer256(non_makeup_img,makeup_img)
    return transfer_img

example = {}
non_makeup_dir = 'examples/non_makeup'
makeup_dir = 'examples/makeup'
non_makeup_list = [os.path.join(non_makeup_dir, file) for file in os.listdir(non_makeup_dir)]
non_makeup_list.sort()
makeup_list = [os.path.join(makeup_dir, file) for file in os.listdir(makeup_dir)]
makeup_list.sort()

def main():
    base = cv2.imread('examples/base.png')
    base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    joker = cv2.imread('examples/joker.png')
    joker = cv2.cvtColor(joker, cv2.COLOR_BGR2RGB)
    halloween = cv2.imread('examples/halloween.png')
    halloween = cv2.cvtColor(halloween, cv2.COLOR_BGR2RGB)
    for source_filepath, makeup_filepath in zip(non_makeup_list, makeup_list):
        source = cv2.imread(source_filepath)
        makeup = cv2.imread(makeup_filepath)

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        makeup = cv2.cvtColor(makeup, cv2.COLOR_BGR2RGB)

        result = get_makeup_transfer_results256(source, joker)
        result_inverse = get_makeup_transfer_results256(result, source)
        fig, ax = plt.subplots(1, 4, figsize=(8, 6))
        
        ax[0].imshow(source)
        ax[1].imshow(joker)
        ax[2].imshow(result)
        ax[3].imshow(result_inverse)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()