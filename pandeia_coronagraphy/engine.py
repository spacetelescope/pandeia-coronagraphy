from __future__ import absolute_import

# Just build an actual subclass of the necessary JWST classes

from copy import deepcopy
from glob import glob
import json
import multiprocessing as mp
import os
import pkg_resources
import sys
import warnings
import astropy.units as units
import astropy.io.fits as fits
from poppy import poppy_core
from scipy.ndimage import convolve
from skimage import draw

if sys.version_info > (3, 2):
    from functools import lru_cache
else:
    from functools32 import lru_cache

import numpy as np

import pandeia
from pandeia.engine.instrument_factory import InstrumentFactory
from pandeia.engine.psf_library import PSFLibrary
from pandeia.engine.psf_library import PSFLibrary as PandeiaPSFLibrary
from pandeia.engine.perform_calculation import perform_calculation as pandeia_calculation
from pandeia.engine.observation import Observation
pandeia_seed = Observation.get_random_seed
from pandeia.engine.astro_spectrum import ConvolvedSceneCube
PandeiaConvolvedSceneCube = ConvolvedSceneCube
from pandeia.engine.constants import SPECTRAL_MAX_SAMPLES
default_SPECTRAL_MAX_SAMPLES = SPECTRAL_MAX_SAMPLES
from pandeia.engine.etc3D import DetectorSignal
PandeiaDetectorSignal = DetectorSignal

try:
    import webbpsf
except ImportError:
    pass

from .pandeia_subclasses import CoronagraphyPSFLibrary, CoronagraphyConvolvedSceneCube, CoronagraphyDetectorSignal
from .config import EngineConfiguration
from . import templates
from . import analysis
# from .templates import templates

# Initialize the engine options
options = EngineConfiguration()

latest_on_the_fly_PSF = None
cache_maxsize = 256     # Number of monochromatic PSFs stored in an LRU cache
                        # Should speed up calculations that involve modifying things
                        # like exposure time and don't actually require calculating new PSFs.


def get_template(filename):
    ''' Look up a template filename.
    '''
    return pkg_resources.resource_filename(templates.__name__,filename)

def list_templates():
    '''
    List all bundled template calculation files.
    '''
    templatewildcard = pkg_resources.resource_filename(templates.__name__, '*.json')
    return [os.path.basename(fname) for fname in glob(templatewildcard)]

def load_calculation(filename):
    with open(filename) as f:
        calcfile = json.load(f)
    return calcfile

def save_calculation(calcfile,filename):
    with open(filename, 'w+') as f:
        json.dump(calcfile, f, indent=2)

def save_to_fits(array,filename):
    hdu = fits.PrimaryHDU(array)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename)

def get_options():
    '''
    This returns the options object, and is used to let the various Pandeia-based subclasses get
    the options object currently in use.
    '''
    return options

def calculate_batch(calcfiles,nprocesses=None):
    if nprocesses is None:
        nprocesses = mp.cpu_count()
    pool = mp.Pool(processes = nprocesses)
    results = pool.map(perform_calculation, calcfiles)
    pool.close()
    pool.join()

    np.random.seed(None) # reset Pandeia seed

    return results

def perform_calculation(calcfile):
    '''
    Manually decorate pandeia.engine.perform_calculation to circumvent
    pandeia's tendency to modify the calcfile during the calculation.

    Updates to the saturation computation could go here as well.
    '''
    if options.on_the_fly_PSFs:
        pandeia.engine.psf_library.PSFLibrary = CoronagraphyPSFLibrary
        pandeia.engine.instrument.PSFLibrary = CoronagraphyPSFLibrary
        pandeia.engine.astro_spectrum.ConvolvedSceneCube = CoronagraphyConvolvedSceneCube
        pandeia.engine.etc3D.DetectorSignal = CoronagraphyDetectorSignal
    else:
        pandeia.engine.psf_library.PSFLibrary = PandeiaPSFLibrary
        pandeia.engine.instrument.PSFLibrary = PandeiaPSFLibrary
        pandeia.engine.astro_spectrum.ConvolvedSceneCube = PandeiaConvolvedSceneCube
        pandeia.engine.etc3D.DetectorSignal = PandeiaDetectorSignal
    if options.pandeia_fixed_seed:
        pandeia.engine.observation.Observation.get_random_seed = pandeia_seed
    else:
        pandeia.engine.observation.Observation.get_random_seed = random_seed

    calcfile = deepcopy(calcfile)
    aperture_dict = CoronagraphyPSFLibrary.parse_aperture(calcfile['configuration']['instrument']['aperture'])
    calcfile['configuration']['instrument']['mode'] = aperture_dict[5]
    calcfile['calculation']['noise'] = options.noise
    calcfule['calculation']['effects'] = options.effects
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category = np.VisibleDeprecationWarning) # Suppress float-indexing warnings
        results = pandeia_calculation(calcfile)

    # Reset the fixed seed state set by the pandeia engine
    # to avoid unexpected results elsewhere
    np.random.seed(None) 

    return results

def random_seed(self):
    '''
    The pandeia engine sets a fixed seed of 42.
    Circumvent that here.
    '''
    #np.random.seed(None) # Reset the seed if already set
    #return np.random.randint(0, 2**32 - 1) # Find a new one
    return None

def calculate_subtracted(aperture_name, target_scene, reference_scene, sgd=False, stepsize=20.e-3):
    """
    This is a function to calculate subtracted images with an optional reference image 
    small-grid dither (SGD). It does the following:
        - Load one of the pandeia_coronagraphy custom templates
        - [optional] construct SGD
        - observe target scene
        - for each reference observation:
            - observe reference scene
        - centre reference to targets
        - create klip PSF
        - subtract klip PSF from target image
        - return subtracted image
    It will return:
        - target image
        - list of reference images
        - artificial PSF
        - subtracted image
    
    Parameters
    ----------
    aperture_name; string
        Name of the convolution aperture to use (this will be used to generate the Pandeia
        calculation other than the scene)
    target_scene: list of dict
        List of Pandeia-style scene dictionaries describing the target
    reference_scene: list of dict
        List of Pandeia-style scene dictionaries describing the reference source
    iterations: int, default=1
        Number of times to iterate generating TA errors and observing the target and reference
        source

    Returns
    -------
    target: numpy array containing <iterations> detector slopes of the target
    references: list of numpy arrays containing <iterations> detector slopes of the reference
    psf: artificial PSF created with klip
    subtracted: numpy array containing <iterations> reference-subtracted target images
    """
    from .scene import create_SGD, get_ta_error, offset_scene
    from .analysis import klip_projection, register_to_target
    aperture_reference = CoronagraphyPSFLibrary.parse_aperture(aperture_name)
    image_mask, pupil_mask, fov_pixels, trim_fov_pixels, pix_scl, mode = aperture_reference
    if mode == 'imaging':
        instrument = 'miri'
    else:
        instrument = 'nircam'
    
    target = load_calculation(get_template('{}_coronagraphy_template.json'.format(instrument)))
    target['scene'] = target_scene
    reference = deepcopy(target)
    reference['scene'] = reference_scene
    
    # add a unique TA error for the target
    errx, erry = get_ta_error()
    offset_scene(target['scene'], errx, erry)

    # add a unique TA error for the reference
    errx_ref, erry_ref = get_ta_error()
    offset_scene(reference['scene'], errx_ref, erry_ref)
    
    if sgd:
        sgds = create_SGD(reference, stepsize=stepsize)
    else:
        sgds = create_SGD(reference, stepsize=stepsize, pattern_name="SINGLE-POINT")

    sgd_results = []
    for sgd in sgds:
        sgd_results.append(perform_calculation(sgd))

    targ_results = perform_calculation(target)

    target_slope = targ_results['2d']['detector']

    sgd_reg = []
    sgd_slopes = []
    for r in sgd_results:
        slope = r['2d']['detector']
        sgd_slopes.append(slope)
        reg = register_to_target(slope, target_slope, rescale_reference=True)
        sgd_reg.append(reg)

    sgd_reg = np.array(sgd_reg)

    centered_target = target_slope - np.nanmean(target_slope)
    artificialPSF = klip_projection(centered_target,sgd_reg)

    sgd_sub = centered_target - artificialPSF
    
    output =    {
                    'target': target_slope,
                    'references': sgd_slopes,
                    'psf': artificialPSF,
                    'subtracted': sgd_sub
                }

    return output

def calculate_contrast_profile(aperture_name, target_scene, reference_scene, iterations=1):
    """
    This is a replacement for the Pandeia calculate_contrast function. It is designed to use the
    various internal analysis functions to do the following:
        - Load one of the pandeia_coronagraphy custom templates
        - Repeat <iterations> times:
            - Run through pandeia with the target scene in place of the default scene
            - Run through pandeia with the reference scene in place of the default scene
        - Run through pandeia with the off-axis target
        - Generate an aperture image
        - Run the analysis contrast utility method
    It will return:
        - list of target images
        - list of reference images
        - off-axis image
        - list of subtracted images
        - normalized contrast profile (with reference bins)
    
    Parameters
    ----------
    aperture_name; string
        Name of the convolution aperture to use (this will be used to generate the Pandeia
        calculation other than the scene)
    target_scene: list of dict
        List of Pandeia-style scene dictionaries describing the target
    reference_scene: list of dict
        List of Pandeia-style scene dictionaries describing the reference source
    iterations: int, default=1
        Number of times to iterate generating TA errors and observing the target and reference
        source

    Returns
    -------
    targets: list of numpy arrays containing <iterations> detector slopes of the target
    references: list of numpy arrays containing <iterations> detector slopes of the reference
    unocculted: numpy array of the detector slopes of the unocculted source
    subtractions: list of numpy arrays containing <iterations> reference-subtracted target images
    contrast: list of numpy arrays containing:
        bins: bins used for the normalized contrast profile
        contrast: normalized contrast profile
    """
    from .scene import get_ta_error, offset_scene
    aperture_reference = CoronagraphyPSFLibrary.parse_aperture(aperture_name)
    image_mask, pupil_mask, fov_pixels, trim_fov_pixels, pix_scl, mode = aperture_reference
    if mode == 'imaging':
        instrument = 'miri'
    else:
        instrument = 'nircam'
    
    target = load_calculation(get_template('{}_coronagraphy_template.json'.format(instrument)))
    target['scene'] = target_scene
    reference = deepcopy(target)
    reference['scene'] = reference_scene
    
    n_observations = iterations
    target_slopes = []
    reference_slopes = []
    for n in range(n_observations):
        if options.verbose:
            print("Iteration {} of {}".format(n+1, n_observations))
        # Add unique target acq error to the target
        current_target = deepcopy(target)
        offset_scene(current_target['scene'], *get_ta_error() )
        # Add unique target acq error to the reference
        current_reference = deepcopy(reference)
        offset_scene(current_reference['scene'], *get_ta_error() )
        # Adopt a new realization of the WFE.
        # Note that we're using the same WFE for target and reference here.
        options.on_the_fly_webbpsf_opd = ('OPD_RevW_ote_for_NIRCam_predicted.fits', n)
        # Calculate target and reference
        targcalc = perform_calculation(current_target)
        target_slopes.append(targcalc['2d']['detector'])
        refcalc = perform_calculation(current_reference)
        reference_slopes.append(refcalc['2d']['detector'])

    offaxis = load_calculation(get_template('{}_coronagraphy_template.json'.format(instrument)))
    offaxis['calculation']['effects']['saturation'] = False # Disable saturation
    offset_scene(offaxis['scene'], 0.5, 0.5) #arsec

    offaxis_slope = perform_calculation(offaxis)['2d']['detector']

    subtraction_stack = np.zeros((n_observations,) + target_slopes[0].shape)
    for i, (targ, ref) in enumerate(zip(target_slopes, reference_slopes)):
        aligned_ref = analysis.register_to_target(ref, targ) # Aligned, scaled, mean-centered reference
        subtraction_stack[i] = targ - np.nanmean(targ) - aligned_ref # Mean-center target and subtract reference

    cov_matrix = analysis.covariance_matrix(subtraction_stack)

    image_dim = subtraction_stack[0].shape
    radius = 5
    aperture_image = np.zeros(image_dim)
    aperture_image[draw.circle((image_dim[0] - 1) // 2, (image_dim[1] - 1) // 2, radius)] = 1

    bins, normalized_profile = analysis.compute_contrast(subtraction_stack, offaxis_slope, aperture_image)
    
    output =    {
                    'targets': target_slopes,
                    'references': reference_slopes,
                    'unocculted': offaxis_slope,
                    'subtractions': subtraction_stack,
                    'contrast': {
                                    'bins': bins,
                                    'profile': normalized_profile
                                }
                }

    return output
