import cv2
import numpy as np
import numpy.linalg as la
from util import read_images

TOLERANCE = 10e-5


def convert_to_lab(images):
    return [cv2.cvtColor(image, cv2.COLOR_BGR2LAB) for image in images]


def get_channel_data(images):
    height, width, channel = images[0].shape
    channel_data = []

    for i in xrange(channel):
        data = []
        for image in images:
            pixels = image[:, :, i]
            pixels = pixels.flatten()
            data.append(pixels)

        data = np.transpose(data)
        channel_data.append(data)

    return channel_data


def init_lagrange_multiplier(data, lmbda):
    Y = np.sign(data)
    norm_two = la.norm(Y, ord=2)
    norm_inf = la.norm(Y, ord=np.inf) / lmbda
    factor = max(norm_two, norm_inf)
    return Y / factor


def shrink(data, tau):
    return np.sign(data) * np.maximum((np.abs(data) - tau), np.zeros(data.shape))


def svd_shrink(data, tau):
    U, S, V = la.svd(data, full_matrices=False)
    return np.dot(U, np.dot(np.diag(shrink(S, tau)), V))


def compute_robust_pca(D):
    rho = 1.5
    mu = 0.5 / la.norm(D, ord=2)
    lmbda = 1 / np.sqrt(np.max(D.shape))

    norm_D = la.norm(D)
    proj_tolerance = 10e-6 * norm_D

    Y = init_lagrange_multiplier(D, lmbda)
    A = np.zeros(D.shape)
    E = np.zeros(D.shape)

    outer_converged = False

    while not outer_converged:
        inner_converged = False
        while not inner_converged:

            temp_A = svd_shrink(D - E + Y / mu, 1 / mu)
            temp_E = shrink(D - A + Y / mu, lmbda / mu)

            norm_A = la.norm(A - temp_A)
            norm_E = la.norm(E - temp_E)

            if norm_A < proj_tolerance and norm_E < proj_tolerance:
                inner_converged = True

            A = temp_A
            E = temp_E

        Z = D - A - E
        Y = Y + mu * Z
        mu *= rho

        norm_Z = la.norm(Z) / norm_D

        if norm_Z < TOLERANCE:
            outer_converged = True

        print "norm_Z: ", norm_Z, " rank: ", la.matrix_rank(A)

    print
    return A, E


def get_background(input_folder):
    images, names = read_images(input_folder)
    channel_data = get_channel_data(images)
    image_shape = images[0].shape
    num_of_images = len(images)
    num_of_channels = len(channel_data)
    max_rank = 0

    print "names:\n", names
    backgrounds = []

    for i in xrange(num_of_channels):
        data = channel_data[i]
        channel_background, channel_reflection = compute_robust_pca(data)
        background_rank = la.matrix_rank(channel_background)
        max_rank = max(max_rank, background_rank)

        if background_rank == 1:
            mean_background = np.mean(channel_background, axis=1).reshape(-1, 1)
            channel_background = np.repeat(mean_background, num_of_images, axis=1)

        # channel_background = channel_background.reshape(image_shape[:2])
        # background[:, :, i] = channel_background
        backgrounds.append(channel_background)
        print "background rank: ", background_rank

    results = []
    backgrounds = np.array(backgrounds)
    print "backgrounds shape: ", backgrounds.shape
    for i in xrange(num_of_images):
        image = np.zeros(image_shape)
        for j in xrange(num_of_channels):
            data = backgrounds[j, :, i].reshape(image_shape[:2])
            image[:, :, j] = data

        results.append(image)

        if max_rank == 1:
            break

    return results, names
