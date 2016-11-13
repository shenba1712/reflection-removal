import numpy as np
from util import read_images
from sklearn.cluster import KMeans


def get_background(input_folder, num_of_images="all"):
    images = read_images(input_folder)
    num_of_images = len(images) if num_of_images == "all" else num_of_images
    image_shape = images[0].shape
    background = np.zeros(image_shape)

    classifier = KMeans(n_clusters=2)
    for i in np.ndindex(image_shape[:2]):
        pixels = []

        for j in xrange(num_of_images):
            pixels.append(images[j][i])
        classifier.fit(pixels)

        count = np.array([0, 0])
        for k in classifier.labels_:
            count[k] += 1

        color = classifier.cluster_centers_[np.argmax(count)]
        background[i] = np.round(color)

    return background
