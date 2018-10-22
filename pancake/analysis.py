from __future__ import absolute_import

from itertools import product

import numpy as np
from scipy.ndimage import convolve

from .transformations import align_fourierLSQ, fourier_imshift

def get_klip_basis(R, cutoff):
    '''
    Succinct KLIP implementation courtesy of N. Zimmerman
    '''
    w, V = np.linalg.eig(np.dot(R, np.transpose(R)))
    sort_ind = np.argsort(w)[::-1] #indices of eigenvals sorted in descending order
    sv = np.sqrt(w[sort_ind]).reshape(-1,1) #column of ranked singular values
    Z = np.dot(1./sv*np.transpose(V[:, sort_ind]), R)
    return Z[0:cutoff, :], sv

def klip_projection(target,reflib,truncation=10):
    refflat = reflib.reshape(reflib.shape[0],-1)
    targflat = target.flatten()
    Z, _ = get_klip_basis(refflat,truncation)
    proj = targflat.dot(Z.T)
    return Z.T.dot(proj).reshape(target.shape)

def register_to_target(reference_image,target_image,mask=None,rescale_reference=True,return_fit=False):
    '''
    Given a reference PSF and a target image, determine the misalignment
    between the two and shift the reference onto the target.

    Parameters:
        reference_image : nd array
            2D image to align to the target_image
        target_image : nd array
            2D image to which the reference_image will be aligned
        mask : nd array, optional
            Mask to weight the fitting process.
        rescale_reference: bool, optional
            Rescale the reference_image as well as align it?
        return_fit : bool, optional
            Return the best-fit offset in (x,y) as well as the scale

    Returns:
        registered_ref : nd array
            referece_image shifted (and scaled, if requested) onto
            the target_image frame
        offx, offy, scale : floats
            The direction of the reference offset. In other words,
            the reference_image would need to be shifted in the
            opposite direction of (offx,offy) and divided by
            scale to recover the best fit to the
            target.
    '''
    centered_ref = reference_image - np.nanmean(reference_image)
    centered_targ = target_image - np.nanmean(target_image)
    offx, offy, scale = align_fourierLSQ(centered_targ, centered_ref, mask)
    if rescale_reference:
        registered_ref = fourier_imshift(centered_ref, -offx, -offy) / scale
    else:
        registered_ref = fourier_imshift(centered_ref, -offx, -offy)

    if return_fit:
        return registered_ref, offx, offy, scale
    else:
        return registered_ref

def compute_contrast(data_stack, offaxis_image, aperture):
    ''' Compute the contrast curve for a stack of data via
    a covariance matrix approach.

    Parameters:
        data_stack : nd array
            Z x Y x X stack of images
        offaxis_image : nd array
            Y x X image of an unocculted source.
        aperture : nd array
            2D image of desired aperture in which
            to compute correlated noise. Should be
            the same dimensions of each image in the
            data stack.

    Returns:
        bins : nd array
            Radial separation (in pixels) at which
            the contrast curve is evaluated
        profile : nd array
            The contrast at the bin radii
    '''
    image_dim = data_stack[0].shape

    # Get the covariance
    cov_matrix = covariance_matrix(data_stack)

    # Get the aperture matrix
    ap_matrix = aperture_matrix(aperture)

    # Find correlated noise within the aperture
    noise = noise_map(cov_matrix, ap_matrix, image_dim)

    # Convolve off-axis source with aperture and take max
    convolved_offaxis = convolve(offaxis_image, aperture, mode='constant')
    normalization = convolved_offaxis.max()

    # Compute the radial profile of the noise map
    bins, profile = radial_profile(noise)

    return bins, profile / normalization

def covariance_matrix(data_stack, mean_subtract=False):
    '''
    Given a cube of images, compute the pixel-wise covariance matrix

    Parameters:
        data_stack : array-like
            Z x Y x X cube of images, where (X, Y) are image dimensions
        mean_subtract: bool, opt.
            Compute the mean centered covariance? By default, we set this
            to False, since--after many discussions--it's been decided
            that removing the mean does something similar to removing a
            reference PSF, which is undesirable for a raw contrast calculation,
            for example.

    Returns:
        XxY x XxY covariance matrix.
    '''
    nstack = data_stack.shape[0]

    # Flatten (z, y, x) to (y*x, z)
    flat = data_stack.reshape(nstack, -1).T

    if mean_subtract:
        mean = np.array([np.mean(flat, axis = 1)] * nstack).T
        flat -= mean

    covariance = flat.dot(flat.T) / (nstack - 1)

    return covariance

def aperture_matrix(aperture):
    '''
    Construct an aperture matrix that mirrors the structure
    of the covariance matrix. It represents the flattened
    aperture centered at each pixel position.

    Parameters:
        aperture : nd array
            The aperture kernel centered in
            an array of the desired image.
    Returns:
        ap_matrix : nd array
            An NxK x NxK array representing the aperture centered at each pixel.
    '''
    # Embed the kernel in an oversized array
    dim = aperture.shape
    desired = (max(dim) * 2 + 1) / 2
    add_x = int(np.floor((desired - dim[0] // 2)))
    add_y = int(np.floor((desired - dim[1] // 2)))
    add = ((add_x, add_x), (add_y, add_y))
    kernel = np.pad(aperture, add, mode = 'constant')
    
    # Loop over every (y, x) pixel shift and record the flattened aperture
    shape = aperture.shape    
    ap_matrix = np.zeros((shape[0] * shape[1], shape[0] * shape[1]))
    for i, (y, x) in enumerate(product(range(shape[0]), range(shape[1]) )):
        ap_matrix[i] = kernel[shape[0] - y : 2 * shape[0] - y,
                                    shape[1] - x : 2 * shape[1] - x].flatten()
    return ap_matrix

def noise_map(cov_matrix, ap_matrix, image_dim):
    ''' From a covariance matrix and the corresponding
    aperture matrix, return the correlated noise within
    the aperture at each pixel.

    Parameters:
        covariance_matrix : nd array
            NxK x NxK covariance matrix. See analysis.covariance_matrix
        ap_matrix : nd array
            NxK x NxK aperture matrix. See analysis.aperture_matrix
        image_dim : tuple
            (N, K) dimensions of original image (and returned noise map)

    Returns:
        noise : np array
            N x K array of the standard deviation at each pixel
    '''
    noise_matrix = ap_matrix.dot(cov_matrix.dot(ap_matrix.T))
    noise = np.sqrt(np.diag(noise_matrix).reshape(image_dim))
    return noise

def radial_profile(image):
    ''' Find the radial profile of an image.

    Parameters:
        image : nd array
            2D image over which to find profile

    Returns:
        bins : nd array
            Bins in which radial mean is computed
        profile : nd array
            Mean value of image in each radial bin
    '''
    # Compute radial distance from center (in pixels) and discretize
    indices = np.indices(image.shape)
    center = np.array(image.shape) / 2.
    radial = np.sqrt( (indices[0] - center[0])**2 + (indices[1] - center[1])**2 ).astype(int)
    # Take mean of image within discretized radial points
    profile = np.bincount(radial.ravel(), image.ravel()) / np.bincount(radial.ravel())
    # Return unique bins, profile
    bins = np.unique(radial)
    return bins, profile
