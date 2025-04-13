import os

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import cv2


def augment_images(image, flip_prob=1, blur_prob=1, contrast_prob=1):
    """
        Применяет аугментации к изображению: горизонтальное отражение, размытие, изменение контраста.

        Параметры:
        - image: PIL.Image или numpy.ndarray - исходное изображение.
        - flip_prob: float - вероятность горизонтального отражения (от 0 до 1).
        - blur_prob: float - вероятность применения размытия (от 0 до 1).
        - contrast_prob: float - вероятность изменения контраста (от 0 до 1).

        Возвращает:
        - PIL.Image - аугментированное изображение.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    if random.random() <= flip_prob:
        flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() <= blur_prob:
        img_np = np.array(image)
        kernel_size = random.choice([3, 5, 7])
        img_np = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        blur_image = Image.fromarray(img_np)

    if random.random() <= contrast_prob:
        enhancer = ImageEnhance.Contrast(image)
        contrast_factor = random.uniform(0.5, 0.5)
        cont_image = enhancer.enhance(contrast_factor)

    return flip_image, blur_image, cont_image

def main():
    img_folder = "train/images"
    mask_folder = "train/masks"

    image_paths = [os.path.join(img_folder, f) for f in os.listdir(img_folder)]
    mask_paths = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder)]
    image_names = [os.path.basename(image_path) for image_path in image_paths]
    mask_names = [os.path.basename(mask_path) for mask_path in mask_paths]

    for img in image_names:
        image = Image.open(img_folder + "/" + img)
        mask = Image.open(mask_folder + "/" + os.path.splitext(img)[0] + "_mask.png")
        flip_image, blur_image, cont_image = augment_images(image)

        blur_image.save(img_folder + "/" + os.path.splitext(img)[0] + "_blur.jpg")
        mask.save(mask_folder + "/" + os.path.splitext(img)[0] + "_blur_mask.png")

        cont_image.save(img_folder + "/" + os.path.splitext(img)[0] + "_cont.jpg")
        mask.save(mask_folder + "/" + os.path.splitext(img)[0] + "_cont_mask.png")

        flip_image.save(img_folder + "/" + os.path.splitext(img)[0] + "_flip.jpg")
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        mask.save(mask_folder + "/" + os.path.splitext(img)[0] + "_flip_mask.png")


if __name__ == "__main__":
    main()