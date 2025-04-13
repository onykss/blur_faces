import argparse
import logging
import numpy as np
from video_processor import VideoProcessor
from frame_processor import FrameProcessor
from object_detector import ObjectDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("blur_face.log"),
        logging.StreamHandler()
    ]
)


def main():
    batch_size = 16
    arg_parser = argparse.ArgumentParser(description="Цензура лиц на видеоролике")
    arg_parser.add_argument("--filename", required=True, help="Название видеофайла (должен лежать в папке Input)")
    args = arg_parser.parse_args()

    video_proc = VideoProcessor(args.filename)

    frames = video_proc.read_video()
    logging.info("Видео загружено")
    rgb_frames = video_proc.get_rgb(frames)
    logging.debug("Кадры преобразованы в RGB")
    attrs = video_proc.get_attrs()
    obj_detector = ObjectDetector()
    logging.info("Модель загружена")
    logging.info("Начинается обработка кадров...")

    batchs = batchify(rgb_frames, batch_size)
    for batch in batchs:
        frame_proc = FrameProcessor(batch, attrs)
        processed_batch = frame_proc.hide_face(obj_detector)
        video_proc.write_batch(np.array(processed_batch))

    video_proc.release_writer()
    logging.info(f"Сборка видео завершена, результат доступен по пути: output/{args.filename.split('.')[0]}.mp4")


def batchify(data, batch_size):
    """
    Разделяет массив данных на батчи заданного размера.
    :param data: Список или массив данных.
    :param batch_size: Размер батча.
    :return: Генератор батчей.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


if __name__ == "__main__":
    main()