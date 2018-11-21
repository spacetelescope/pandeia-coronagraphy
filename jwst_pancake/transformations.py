
import numpy as np
from scipy import optimize
from scipy.ndimage import fourier_shift

def cart_to_polar(xy):
    '''convert separations into offset, theta'''
    r = np.sqrt(xy[0]**2 + xy[1]**2) #radius, arcsec
    theta = np.rad2deg(np.arctan2(xy[1],xy[0])) #angle, degrees
    return r, theta

def polar_to_cart(r,theta):
    '''convert from polar to cartesian,
    assuming theta in degrees'''
    x = r * np.cos(np.deg2rad(theta))
    y = r * np.sin(np.deg2rad(theta))
    
    return np.array([x, y])

def affine_transform(theta,center):
    ''' theta = deg
    center = x, y
    '''
    theta_rad = np.deg2rad(theta)
    x, y = center
    a = np.cos(theta_rad)
    b = np.sin(theta_rad)
    M = np.array([[a, b, (1-a)*x - b*y],
                  [-b, a, b*x + (1-a)*y],
                  [0,0,1]])
    return M

def rotate(xy,theta,center):
    M = affine_transform(theta,center)
    x,y = xy
    return np.dot(M,[x,y,1])

def align_fourierLSQ(reference,target,mask=None):
    '''LSQ optimization with Fourier shift alignment

    Parameters:
        reference : nd array
            N x K image to be aligned to
        target : nd array
            N x K image to align to reference
        mask : nd array, optional
            N x K image indicating pixels to ignore when
            performing the minimization. The masks acts as
            a weighting function in performing the fit.

    Returns:
        results : list
            [x, y, beta] values from LSQ optimization, where (x, y) 
            are the misalignment of target from reference and beta
            is the fraction by which the target intensity must be
            reduced to match the intensity of the reference.
    '''

    init_pars = [0.,0.,1.]
    out,_ = optimize.leastsq(shift_subtract, init_pars, args=(reference,target,mask))
    results = [out[0],out[1],out[2]] #x,y,beta
    return results

def shift_subtract(params,reference,target,mask=None):
    '''Use Fourier Shift theorem for subpixel shifts.

    Parameters:
        params : tuple
            xshift, yshift, beta
        reference : nd array
            See align_fourierLSQ
        target : nd array
            See align_fourierLSQ
        mask : nd array, optional
            See align_fourierLSQ

    Returns:
        1D nd array of target-reference residual after
        applying shift and intensity fraction.
    '''
    xshift, yshift, beta = params
    
    offset = fourier_imshift(reference,xshift,yshift)
    
    if mask is not None:
        return ( (target - beta * offset) * mask ).flatten()
    else:
        return ( target - beta * offset ).flatten()

def fourier_imshift(image,xshift,yshift):
    '''  Shift an image by use of Fourier shift theorem

    Parameters:
        image : nd array
            N x K image
        xshift : float
            Pixel value by which to shift image in the x direction
        yshift : float
            Pixel value by which to shift image in the y direction

    Returns:
        offset : nd array
            Shifted image

    '''
    offset = fourier_shift( np.fft.fftn(image), (-yshift,xshift) )
    offset = np.fft.ifftn(offset).real
    return offset