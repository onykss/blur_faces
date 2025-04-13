import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


class ObjectMasker:
    def __init__(self, batch, masks):
        self.batch = batch
        self.masks = masks

    def mask_object(self, frame, mask):
        numpy_frame = np.array(frame)
        numpy_mask = np.array(mask)

        assert numpy_frame.shape[:2] == numpy_mask.shape, "Размеры кадра и маски не совпадают"

        result = numpy_frame.copy()

        for channel in range(3):
            blurred_channel = gaussian_filter(numpy_frame[:, :, channel], sigma=25)
            result[:, :, channel][numpy_mask > 0] = blurred_channel[numpy_mask > 0]

        return Image.fromarray(result)

    def mask_objects_in_batch(self):
        return [self.mask_object(self.batch[i], self.masks[i]) for i in range(len(self.batch))]
