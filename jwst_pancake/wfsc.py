# Wavefront Sensing & Control Related PSF Calculations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import astropy
import astropy.io.fits as fits
import webbpsf
import poppy


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
    refstar['spectrum']['normalization']['bandpass'] = 'bessell,k'
    refstar['spectrum']['normalization']['norm_flux'] = kmag
    refstar['spectrum']['normalization']['norm_fluxunit'] = 'vegamag'
    refstar['spectrum']['sed']['key'] = sptype
    refstar['spectrum']['sed']['sed_type'] = 'phoenix'


def configure_telescope(step='mimf', defocus_waves=0.0, jitter_sigma=0.007, jitter='gaussian'):
    """ Convenience function for configuring the telescope

    defocus_waves : waves of defocus, at 2 microns wavelength
    jitter, jitter_sigma : parameters for jitter model. See WebbPSF docs.
    """
    if not engine.options.on_the_fly_PSFs:
        engine.options.on_the_fly_PSFs =  True
        print("Enabling on-the-fly PSFs for ETC calculations")

    if step.lower()=='mimf':
        engine.options.on_the_fly_webbpsf_options['defocus_waves'] = defocus_waves
        engine.options.on_the_fly_webbpsf_options['defocus_wavelength'] = 2e-6
    else:
        raise ValueError("Don't know how to configure for step={} yet".format(step))

    engine.options.on_the_fly_webbpsf_options['jitter'] = jitter
    engine.options.on_the_fly_webbpsf_options['jitter_sigma'] = jitter_sigma


def configure_readout(calc_input, ngroup=10, nint=1, nexp=1, readout_pattern='rapid'):
    """ Convenience function for configuring the detector readout."""
    calc_input['configuration']['detector']['ngroup'] = ngroup
    calc_input['configuration']['detector']['nint'] = nint
    calc_input['configuration']['detector']['nexp'] = nexp
    if readout_pattern is not None:
        calc_input['configuration']['detector']['readout_pattern'] = readout_pattern


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


def autosize_hex_mask(results):
    """Generate a hexagon mask scaled to include most of the
    flux in an image, defined as on average including pixels
    above 1/8th of the peak count. This is a heuristic hack
    that is designed to approximate what Acton said he did for
    making SBR masks: "Look at a cut across the PSF and see
    where it falls down sharply from the peak; it's pretty
    abrupt."

    """
    # flatten image, compare to threshold
    colsum = np.sum(results['2d']['detector'],axis=0)
    thresh=colsum.max()/4
    pixscale = results['psf']['pix_scl']* results['psf']['upsamp']
    try:
        boxwidth = (np.max(np.where(colsum>thresh))-np.min(np.where(colsum>thresh)))*pixscale
    except:
        import warnings
        warnings.warn("Unable to auto-size box width. Falling back to 10 pix default.")
        boxwidth = 10*pixscale

    # generate a hexagon mask that matches.
    wf= poppy.Wavefront(pixelscale=pixscale,npix=results['2d']['detector'].shape[0])
    hexmask = poppy.optics.HexagonFieldStop(flattoflat=boxwidth, rotation=30).get_transmission(wf)
    return hexmask



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
    #starmask = results['2d']['snr']>results['2d']['snr'].max()/4
    starmask = autosize_hex_mask(results) > 0

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
    inttime_tot = results['information']['exposure_specification']['saturation_time'] # this is across all ints
    instr =  engine.InstrumentFactory(results['input']['configuration'])

    nint = results['information']['exposure_specification']['nint']
    inttime = inttime_tot / nint

    det_pars = instr.det_pars
    if instr.inst_name=='nircam':
        # For NIRCam, need to select either SW vs LW saturation value
        try:
            fullwell = det_pars['fullwell'][instr.get_aperture()]
        except KeyError:
            fullwell = det_pars['fullwell']['default']
    elif instr.inst_name=='nirspec':
        fullwell = det_pars['fullwell']['full']
    #    det_pars = det_pars['imager'] # NOT either MRS one!
    else:
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
    {0[detector][readout_pattern]}, ngroups={0[detector][ngroup]}, nints={0[detector][nint]}, nexps={0[detector][nexp]}
Target is K={2[normalization][norm_flux]:.1f}, SpType={2[sed][key]}""".format(
        obsdict['configuration'],
        obsdict['configuration']['instrument']['filter'].upper(),
        obsdict['scene'][0]['spectrum'],
        obsdict['configuration']['instrument']['instrument'].upper()))

def colorbar_setup_helper(nticks=5, label=None, *args, **kwargs):
    cb = plt.colorbar(orientation='horizontal', *args, **kwargs)
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=nticks)
    cb.locator = tick_locator
    cb.update_ticks()
    if label is not None:
        cb.set_label(label)

def display_one_image(image, scale, imagecrop=None, ax=None,**kwargs):
    """ Display an image with axes in arcseconds """

    if ax is None:
        ax = plt.gca()
    halfsize = image.shape[0]/2 *scale
    extent= [-halfsize, halfsize, -halfsize, halfsize]
    im = ax.imshow(image, extent=extent, vmin=0, **kwargs)

    if imagecrop is not None:
        ax.set_xlim(-imagecrop/2, imagecrop/2)
        ax.set_ylim(-imagecrop/2, imagecrop/2)
    return im


def display_mimf_etc_results(results, verbose=True):

    fig = plt.figure(figsize=(20,9))
    plt.subplots_adjust(top=0.8)

    # Display PSF
    ax1=plt.subplot(131)
    display_one_image(results['psf']['int'], results['psf']['pix_scl'])
    plt.title("Oversampled Monochromatic Input PSF \n(WebbPSF on-the-fly)")
    plt.ylabel("Arcsec")
    colorbar_setup_helper(label='Fractional Counts', mappable=ax1.images[0])

    # Display detector image
    ax2=plt.subplot(132)
    display_one_image(results['2d']['detector'], results['psf']['pix_scl']*results['psf']['upsamp'])
    plt.title("Pandeia result: Detector")
    colorbar_setup_helper(label='e-/sec', mappable=ax2.images[0])
    well_fraction, inttime = assess_well_fraction(results)
    ax2.text(0.03, 0.03, "Peak pixel: {:.1f}% of full well\nafter T_int={:.1f} s\nT_total={:.1f} s for {} ints".format(
        well_fraction*100, inttime,
        results['information']['exposure_specification']['exposure_time'],
        results['information']['exposure_specification']['nint']
    ), color='white',
        transform=ax2.transAxes, fontsize=15)

    SBR, mask = calc_sbr(results, return_mask=True)
    results['mask']=mask
    results['SBR']=SBR
    if verbose:
        print(f"SBR: {SBR}")
        print(f"Mean count rate within mask: {(results['2d']['detector'][mask]).mean():.1f} e-/sec")
        print(f"Total count rate: {(results['2d']['detector']).sum():.1f} e-/sec")
        print(f"SBR and mask have been added to results dict")

    # Display SNR
    ax3=plt.subplot(133)
    display_one_image(results['2d']['snr'], results['psf']['pix_scl']*results['psf']['upsamp'])

    halfsize = results['2d']['snr'].shape[0]/2 *results['psf']['pix_scl']*results['psf']['upsamp']
    extent= [-halfsize, halfsize, -halfsize, halfsize]


    ax3.contour(mask, extent=extent, colors='white', linewidths=1, alpha=0.25, linestyles="--")
    plt.title("Pandeia result: SNR")
    colorbar_setup_helper(label='SNR', mappable=ax3.images[0])


    ax3.text(0.03, 0.03, "SBR: {:.1f} inside above mask".format(SBR), color='white',
            transform=ax3.transAxes, fontsize=15)

    description = describe_obs(results['input']) + "\nSBR: {:.2f}".format(SBR)
    plt.suptitle(description, fontsize=16)

    if 'saturated' in results['warnings']:
        plt.text(0.1, 0.1, results['warnings']['saturated'], color='red',
                transform=fig.transFigure, fontsize=18)
    elif 'nonlinear' in results['warnings']:
        plt.text(0.1, 0.1, results['warnings']['nonlinear'], color='darkorange',
                transform=fig.transFigure, fontsize=18)


#### Hacks and Monkey Patches to implement a version of FGS into Pandeia.

import pandeia.engine.jwst, pandeia.engine.instrument_factory
class FGS(pandeia.engine.jwst.JWSTInstrument):
    """ HIGHLY UNOFFICIAL & UNSUPPORTED FGS IMPLEMENTATION
    NOT INTENDED FOR USE OUTSIDE OF STSCI
    THIS IS A HACK -- USE AT YOUR OWN RISK -- BETTER YET, DON'T USE

    FGS has nothing besides an imager, so does not need any functions subclassed.
    """
    @property
    def test_function(self):
        return "The FGS is totally unsupported in Pandeia."

pandeia.engine.jwst.FGS = FGS
pandeia.engine.instrument_factory.FGS = FGS


