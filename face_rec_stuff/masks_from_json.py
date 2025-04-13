import json
import os
from PIL import Image, ImageDraw
import numpy as np


def get_masks_from_json(filename):
    """
    Создает маски в формате png в папку masks из файла .json, полученного с помощью via (утилита для разметки данных)
    :param filename: имя .json файла
    """
    json_file = filename

    output_dir = "masks"
    os.makedirs(output_dir, exist_ok=True)

    image_size = (512, 512)

    with open(json_file, "r") as f:
        data = json.load(f)

    last_file_id = 1
    last_file_name = ""
    points = []

    for metadata_id, annotation in data["metadata"].items():
        file_id = int(annotation["vid"])
        file_name = data["file"][str(file_id)]["fname"]

        if "xy" not in annotation or not annotation["xy"]:
            continue

        if last_file_id != file_id:
            mask = Image.new("L", image_size, 0)
            draw = ImageDraw.Draw(mask)

            for i in points:
                try:
                    draw.polygon(i, outline=255, fill=255)
                except:
                    print(f"Error with {i}")

            mask_file_name = os.path.splitext(last_file_name)[0] + "_mask.png"
            mask.save(os.path.join(output_dir, mask_file_name))
            points = []
            print(f"Создана маска для {last_file_name}")

        xy = annotation["xy"]
        points.append([(round(xy[i]), round(xy[i + 1])) for i in range(1, len(xy), 2)])

        last_file_id = file_id
        last_file_name = file_name

    #создаем маску для последнего элемента
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    for i in points:
        try:
            draw.polygon(i, outline=255, fill=255)
        except :
            print(f"Error with {i}")

    mask_file_name = os.path.splitext(last_file_name)[0] + "_mask.png"
    mask.save(os.path.join(output_dir, mask_file_name))
    points = []
    print(f"Создана маска для {last_file_name}")

if __name__ == "__main__":
    get_masks_from_json("via_project_11Apr2025_15h58m00s.json")

