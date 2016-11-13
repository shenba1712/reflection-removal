import cv2
import numpy as np
import numpy.linalg as la
from util import read_images

TOLERANCE = 10e-5


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


def svd_shrink(data, tau, r=-1):
    U, S, V = la.svd(data, full_matrices=False)
    shrinked_S = np.diag(shrink(S, tau))
    computed_rank = la.matrix_rank(shrinked_S)
    if r != -1 and r < computed_rank:
        S[r:] = 0
        shrinked_S = np.diag(shrink(S, tau))
        print "computed_rank: ", computed_rank
    return np.dot(U, np.dot(shrinked_S, V))


def compute_frpca(D, r=-1):
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

            temp_A = svd_shrink(D - E + Y / mu, 1 / mu, r)
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

        print "norm_Z", norm_Z, " rank: ", la.matrix_rank(A)

    print
    return A, E


def get_background(input_folder, r=-1):
    images, names = read_images(input_folder)
    channel_data = get_channel_data(images)
    image_shape = images[0].shape
    background = np.zeros(image_shape)

    reflection = np.zeros(image_shape)

    for i in xrange(len(channel_data)):
        data = channel_data[i]
        channel_background, channel_reflection = compute_frpca(data, r)
        channel_background = channel_background[:, 0]
        channel_background = channel_background.reshape(image_shape[:2])
        background[:, :, i] = channel_background

        reflection[:, :, i] = channel_reflection[:, 0].reshape(image_shape[:2])

    return background

