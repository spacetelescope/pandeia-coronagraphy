import numpy as np

from transformations import align_fourierLSQ, fourier_imshift

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

def register_to_target(reference_image,target_image):
    centered_ref = reference_image - np.nanmean(reference_image)
    offx, offy, scale = align_fourierLSQ(centered_ref,target_image)
    registered_ref = fourier_imshift(centered_ref,offx,offy) * scale
    return registered_ref