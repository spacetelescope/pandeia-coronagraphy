import numpy as np

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
    #mean center
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
    offx, offy, scale = align_fourierLSQ(centered_targ,centered_ref,mask)
    if rescale_reference:
        registered_ref = fourier_imshift(centered_ref,-offx,-offy) / scale # * scale
    else:
        registered_ref = fourier_imshift(reference_image,-offx,-offy)

    if return_fit:
        return registered_ref, offx, offy, scale
    else:
        return registered_ref