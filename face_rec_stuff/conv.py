from PIL import Image
import os

def resize_images():
    """
    Изменяет размер всех изображений в папке Input
    :return:
    """
    image_height = 512
    imagee_width = 512
    input_dir = "input"

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        try:
            with Image.open(file_path) as img:
                resized_img = img.resize((image_height, imagee_width), Image.Resampling.LANCZOS)

                resized_img.save(file_path)
        except Exception as e:
            print(f"Ошибка при обработке изображения {filename}: {e}")

if __name__ == "__main__":
    resize_images()


