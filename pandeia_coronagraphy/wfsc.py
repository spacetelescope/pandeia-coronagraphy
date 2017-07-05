# Wavefront Sensing & Control Related PSF Calculations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astropy
import astropy.io.fits as fits
import webbpsf


##### Pandeia configuration related functions ####

from . import engine

def configure_star(calc_input, kmag=9, sptype='g2v'):
    """ Convenience function for configuring the star.

    Parameters
    ----------
    kmag : float
        K band apparent magnitude
    sptype : string
        Spectral Type, written like 'g2v'.

    """
    # Define target
    refstar = calc_input['scene'][0]
    refstar['spectrum']['normalization']['type'] = 'photsys'
    refstar['spectrum']['normalization']['bandpass'] = 'bessel,k'
    refstar['spectrum']['normalization']['norm_flux'] = 9
    refstar['spectrum']['normalization']['norm_fluxunit'] = 'vegamag'
    refstar['spectrum']['sed']['key'] = sptype
    refstar['spectrum']['sed']['sed_type'] = 'phoenix'


def configure_telescope(step='mimf', defocus_waves=0.0):
    """ Convenience function for configuring the telescope """
    engine.on_the_fly_PSFs =  True

    if step.lower()=='mimf':
        engine.on_the_fly_webbpsf_options['defocus_waves'] = defocus_waves
        engine.on_the_fly_webbpsf_options['defocus_wavelength'] = 2e-6
    else:
        raise ValueError("Don't know how to configure for step={} yet".format(step))


def configure_readout(calc_input, ngroup=10, nint=1, nexp=1):
    """ Convenience function for configuring the detector readout."""
    calc_input['configuration']['detector']['ngroup'] = ngroup
    calc_input['configuration']['detector']['nint'] = nint
    calc_input['configuration']['detector']['nexp'] = nexp


##### Calculation related functions ####


def get_bg_noise(results):
    """Measure background noise from pixels in the image that are 'far from the star'
    Those background area is inferred as images in the output detector image that are
    beyond the size of the input PSF.

    Returns 1 sigma standard deviation of background pixels

    """
    # Calculate a mask for the pixels which have no stellar signal in them.
    # Array must be large enough!
    shp = results['2d']['detector'].shape
    borderwidth = (shp[0] - (results['psf']['int'].shape[0]/results['psf']['upsamp']))/2

    if borderwidth < 2:
        raise RuntimeError("Output result image size is too small to measure background. "
                           "Increase input['configuration']['scene_size']")

    # Get those pixels
    y, x = np.indices(shp)

    bgmask = ((x < borderwidth) | ((shp[1]-x)<borderwidth) |
                (y < borderwidth) | ((shp[1]-y)<borderwidth))

    # Measure noise
    bgnoise = results['2d']['detector'][bgmask].std()
    return bgnoise

def calc_sbr(results, return_mask=False):
    """ Calc Ball's Surface-to-Background-Ratio (SBR) metric, following Acton's algorithm:

    Take the median value of the pixels with in the image. In the case of a defocused spot, this is just
    the median value within the "top hat "portion of the image. Next, take the standard deviation of the
    pixels that are clearly in the background, that is, have no incident photons on them.
    Take the ratio of these two quantities, and you have the signal to background ratio.
    """
    bgnoise = get_bg_noise(results)

    # WAG: anything with a SNR within a factor of 4 of the peak SNR should
    # count as "in the PSF" for this purpose
    starmask = results['2d']['snr']>results['2d']['snr'].max()/4

    sbr = np.median(results['2d']['detector'][starmask])/bgnoise
    if return_mask:
        return (sbr, starmask)
    else: return sbr

def assess_well_fraction(results, verbose=False):
    """ Assess the peak well depth filled in the detector at the end of the exposure.

    """
    # Note there are several subtle variants of integration time; we want
    # the 'saturation time' which is generally equal to the ramp time
    # unless there are initial dropped frames.
    inttime = results['information']['exposure_specification']['tsat']
    instr =  engine.InstrumentFactory(results['input']['configuration'])

    det_pars = instr.det_pars
    # For NIRCam, need to select either SW vs LW det_par instance
    if instr.inst_name=='nircam':
        det_pars = det_pars[instr.mode[0:2]]
    elif instr.inst_name=='miri':
        det_pars = det_pars['imager'] # NOT either MRS one!

    fullwell = det_pars['fullwell']

    peakpix = results['2d']['detector'].max()*inttime
    fraction = peakpix/fullwell
    if verbose:
        print("Integration time: {:.1f} s\nPeak Pixel: {:.1f} e-\nFull well: {} e-\n  Fraction of full well: {:.3f}".format(
        inttime, peakpix, fullwell, fraction))
    return fraction, inttime



##### Display and plotting functions ####

def describe_obs(obsdict):
    """Produce label text for plot suptitle"""
    return("""{3} observation with {1}.
    {0[detector][readmode]}, ngroups={0[detector][ngroup]}, nints={0[detector][nint]}, nexps={0[detector][nexp]}
Target is K={2[normalization][norm_flux]}, SpType={2[sed][key]}""".format(
        obsdict['configuration'],
        obsdict['configuration']['instrument']['filter'].upper(),
        obsdict['scene'][0]['spectrum'],
        obsdict['configuration']['instrument']['instrument'].upper()))

def colorbar_setup_helper(nticks=5, label=None):
    cb = plt.colorbar(orientation='horizontal')
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=nticks)
    cb.locator = tick_locator
    cb.update_ticks()
    if label is not None:
        cb.set_label(label)

def display_one_image(image, scale, imagecrop=None):
    """ Display an image with axes in arcseconds """
    halfsize = image.shape[0]/2 *scale
    extent= [-halfsize, halfsize, -halfsize, halfsize]
    plt.imshow(image, extent=extent, vmin=0)

    if imagecrop is not None:
        plt.xlim(-imagecrop/2, imagecrop/2)
        plt.ylim(-imagecrop/2, imagecrop/2)

