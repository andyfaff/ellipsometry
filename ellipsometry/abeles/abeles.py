import numpy as np


def snell(n0, n1, theta0):
    return np.arcsin(n0 * np.sin(theta0) / n1)


def reflectance(layers, theta, lam):
    nlayers = np.size(layers, 0) - 2

    layers_n = layers[:, 1] + layers[:, 2]*1j

    thetas = snell(layers_n[0], layers_n, theta)
    kz = 2 * np.pi * layers_n * np.cos(thetas) / lam
