import cv2
import glob
from sklearn.metrics import mean_squared_error


def read_images(input_folder):
    images = []
    names = []
    for path in glob.glob(input_folder + "*.png"):
        name = path.strip().split("/")[-1].replace(".png", "")
        names.append(name)
        images.append(cv2.imread(path, cv2.IMREAD_UNCHANGED))
    return images, names


def compute_mean_square_error(images, ground_truth):
    ground_truth = ground_truth.flatten()
    amount = len(images)
    error_sum = 0.

    for i in xrange(amount):
        image = images[i].flatten()
        error = mean_squared_error(ground_truth, image)
        error_sum += error

    return error_sum / amount
