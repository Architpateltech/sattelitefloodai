import cv2
import numpy as np


class DisasterSegmentation:


    def __init__(self):

        print(
            "Segmentation module ready"
        )


    def create_mask(self, image_path):

        """
        Temporary segmentation.
        Later replaced by U-Net/YOLO segmentation model.
        """


        image = cv2.imread(
            image_path
        )


        height, width, _ = image.shape


        # Empty mask

        mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )


        # Demo affected region
        # center area

        cv2.rectangle(
            mask,
            (100,100),
            (400,400),
            1,
            -1
        )


        return mask