import logging

import cv2
import os


class VideoProcessor():
    def __init__(self, filename):
        self.filename = filename
        self.fps = None
        self.frame_width = None
        self.frame_height = None
        self.output_path = "output/" + filename.split('.')[0] + ".mp4"
        self.video_writer = None

    def read_video(self):
        """
        Считывает видеофайл и разбирает его на кадры
        :return: массив кадров
        """
        video_path = "input/" + self.filename
        if not os.path.exists(video_path):
            logging.error(f"Ошибка: Файл {video_path} не найден.")
        else:
            logging.debug(f"Файл {video_path} найден.")
        cap = cv2.VideoCapture(video_path)

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.frame_width, self.frame_height))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            else:
                frames.append(frame)

        cap.release()
        return frames

    def get_attrs(self):
        """
        :return: Аттрибуты видеофайла
        """
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.fps
        }

    @staticmethod
    def get_rgb(frames):
        return [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    def write_frame(self, frame):
        bgr_frames = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(bgr_frames)

    def release_writer(self):
        self.video_writer.release()

    def write_batch(self, batch):
        for frame in batch:
            self.write_frame(frame)
