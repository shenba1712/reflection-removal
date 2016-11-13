import cv2
import glob
import numpy as np


def get_background(input_folder):
    # Get background
    count = 1
    average_image = None
    for path in glob.glob(input_folder + "*.png"):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if count == 1:
            average_image = np.float32(image)

        alpha = 1./count
        cv2.accumulateWeighted(image, average_image, alpha)
        count += 1

    # Normalise the background image
    norm_image = np.uint8(np.round(average_image))

    return norm_image
