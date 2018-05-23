import numpy as np
import matplotlib.pyplot as plt
import poppy

import pandeia_coronagraphy.transformations


def test_fourier_imshift(plot=False):
    """ Verify we can apply a random shift and the image moves in the
    direction requested """
    size = 512
    center = (size-1)/2
    dx = np.random.randint(-50, 50)
    dy = np.random.randint(-50, 50)
    a0 = poppy.misc.airy_2d(center=(center, center))   # centered PSF we will shift
    a1 = poppy.misc.airy_2d(center=(center+dx, center+dy))  # target for where it should end up

    # Note we have to flip the order of y and x in the call here.
    shifted = pandeia_coronagraphy.transformations.fourier_imshift(a0, dy, dx)

    difference = a1-shifted

    box_min = int(size/2-50)
    box_max = int(size/2+50)

    assert np.abs(difference[box_min:box_max, box_min:box_max]).sum() < 1e-10, "Image difference too large"

    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(141)
        plt.imshow(a0)
        plt.title("Centered\nat ({}, {})".format(center, center))
        plt.subplot(142)
        plt.imshow(a1)
        plt.title("Centered\nat ({}, {})".format(center+dx, center+dy))
        plt.subplot(143)
        plt.imshow(shifted)
        plt.title("Centered, then \n Fourer Shifted by ({}, {})".format(dx, dy))
        plt.subplot(144)
        plt.imshow(a1-shifted)
        plt.title("Diff of \nlast two images".format(dx, dy))



