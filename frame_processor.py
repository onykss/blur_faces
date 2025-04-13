from PIL import Image
from object_detector import ObjectDetector
from object_masker import ObjectMasker
import numpy as np


class FrameProcessor:
    def __init__(self, batch, attrs):
        self.batch = batch
        self.attrs = attrs

    def preprocess_frame(self, frame):
        image_height = 512
        image_width = 512
        pil_frame = Image.fromarray(frame)
        resized_frame = pil_frame.resize((image_width, image_height), Image.Resampling.LANCZOS)
        return resized_frame

    def get_mask(self, obj_detector):
        preprocessed_batch = [self.preprocess_frame(frame) for frame in self.batch]
        masks = obj_detector.detect_face(preprocessed_batch)
        return masks

    def hide_face(self, obj_detector):
        masks = self.get_mask(obj_detector)
        masks = [mask.cpu().numpy() for mask in masks]
        masks = [np.squeeze(mask) for mask in masks]
        masks = [(mask > 0.5).astype(np.uint8) * 255 for mask in masks]
        grey_masks = [Image.fromarray(mask, mode="L") for mask in masks]

        resized_masks = [grey_mask.resize((self.attrs['frame_width'], self.attrs['frame_height']), Image.Resampling.NEAREST)
                         for grey_mask in grey_masks]

        obj_masker = ObjectMasker(self.batch, resized_masks)
        masked_batch = obj_masker.mask_objects_in_batch()
        return masked_batch

