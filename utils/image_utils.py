import cv2
import numpy as np



def create_overlay(image_path, mask):

    image = cv2.imread(
        image_path
    )


    overlay = image.copy()


    overlay[mask == 1] = [
        0,
        0,
        255
    ]


    result = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )


    return result



def create_side_by_side(
        original_path,
        analyzed_image
):


    original = cv2.imread(
        original_path
    )


    height = min(
        original.shape[0],
        analyzed_image.shape[0]
    )


    width = min(
        original.shape[1],
        analyzed_image.shape[1]
    )


    original = cv2.resize(
        original,
        (width,height)
    )


    analyzed_image = cv2.resize(
        analyzed_image,
        (width,height)
    )


    comparison = np.hstack(
        (
            original,
            analyzed_image
        )
    )


    return comparison