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

def calculate_contrast(input, webapp=False):
    """
    This is a replacement for the Pandeia calculate_contrast function. It will only work if called
    from pandeia_coronagraphy with an input file generated via pandeia_coronagraphy. It depends on
    the 'scene' key in the input dictionary being replaced with 2 keys:
        - 'target_scene': contains the target source(s)
        - 'reference_scene': contains the reference source
    In addition, the observing strategy will be set to imaging in order to do single calculations
    for each run (target, reference, unocculted target).
    
    Note that this function returns a report instance in exactly the same way that the pandeia 
    'calculate_contrast' function does, so it's monkey-patched directly into pandeia.
    
    Remaining docstring is from pandeia 'calculate_contrast' function:
    -----
    This is a function to do the 'forward' exposure time calculation where given a dict
    in engine API input format we calculate the resulting coronagraphic contrast and return a Report
    on the results.

    While this method is meant for coronagraphic modes, it will work also for regular imaging modes.

    Parameters
    ----------
    input: dict
        Engine API format dictionary containing the information required to perform the calculation.
    psf_ibrary : psf_library.PSFLibrary instance
        Library of PSF files (e.g. produced by webbpsf) to be used in the calculation

    Returns
    -------
    report.Report instance
    """
    warnings = {}
    try:
        target_scene_configuration = input['target_scene']
        reference_scene_configuration = input['reference_scene']
        background = input['background']
        instrument_configuration = input['configuration']
        strategy_configuration = input['strategy']
        if input.get('debugarrays'):
            debug_utils.init(input.get('debugarrays'))
    except KeyError as e:
        message = "Missing information required for the calculation: %s" % str(e)
        raise EngineInputError(value=message)

    # get the calculation configuration from the input or use the defaults
    if 'calculation' in input:
        calc_config = CalculationConfig(config=input['calculation'])
    else:
        calc_config = CalculationConfig()

    # #### BEGIN calculation #### #
    """
    This section implements the Pandeia engine API.
    """
    # check for empty scene configuration and set it up properly if it is empty.
    if len(scene_configuration) == 0:
        scene_configuration = build_empty_scene()

    instrument = InstrumentFactory(config=instrument_configuration, webapp=webapp)
    warnings.update(instrument.warnings)

    strategy = StrategyFactory(instrument, config=strategy_configuration, webapp=webapp)
    
    # Check for user-specified dithers (the contrast calculation will add an additional 2 fictional dithers).
    if not hasattr(strategy, 'dithers') or len(strategy.dithers) != 1:
        message = "Contrast calculations currently require a single dither " \
                  "to be passed in the strategy, {} was passed".format(strategy.dithers)
        raise EngineInputError(value=message)

    # Create the centred target scene
    target_scene = Scene(input=target_scene_configuration, webapp=webapp)
    if hasattr(strategy, "scene_rotation"):
        target_scene.sources = strategy.rotate(target_scene.sources)
    warnings.update(target_scene.warnings)

    # Create the centred reference scene
    reference_scene = Scene(input=reference_scene_configuration, webapp=webapp)
    if hasattr(strategy, "scene_rotation"):
        reference_scene.sources = strategy.rotate(reference_scene.sources)
    warnings.update(reference_scene.warnings)
    
    # Create the unocculted target scene
    unocculted_scene_configuration = deepcopy(target_scene)
    unocculted_scene = Scene(input=unocculted_scene_configuration, webapp=webapp)
    unocculted_scene.offset({'x': strategy.unocculted_xy[0], 'y': strategy.unocculted_xy[1]})
    if hasattr(strategy, "scene_rotation"):
        unocculted_scene.sources = strategy.rotate(unocculted_scene.sources)
    warnings.update(unocculted_scene.warnings)
    
    obset = []
    for scene in zip(target_scene, reference_scene):
        obset.append(observation.Observation(scene=scene, instrument=instrument, strategy=strategy, background=background, webapp=webapp))

    # seed the random number generator
    seed = obs.get_random_seed()
    np.random.seed(seed=seed)

    # Sometimes there is more than one exposure involved so implement lists for signal and noise
    my_detector_signal_list = []
    my_detector_noise_list = []
    my_detector_saturation_list = []

    for obs in obset:
        # make a new deep copy of the observation for each dither so that each position is offset
        # from the center position. otherwise the offsets get applied cumulatively via the reference.
        o = deepcopy(obs)
        # Calculate the signal rate in the detector plane
        my_detector_signal = DetectorSignal(o, calc_config=calc_config, webapp=webapp)
        my_detector_noise = DetectorNoise(my_detector_signal, o)

        # Every dither has a saturation map
        my_detector_saturation = my_detector_signal.get_saturation_mask()
        my_detector_signal_list.append(my_detector_signal)
        my_detector_noise_list.append(my_detector_noise)
        my_detector_saturation_list.append(my_detector_saturation)

    # We need a regular S/N of the target source
    extracted_sn = strategy.extract(my_detector_signal_list, my_detector_noise_list)
    warnings.update(extracted_sn['warnings'])

    # Use the strategy to get the extracted contrast products
    grid = my_detector_signal_list[0].grid

    aperture = strategy.aperture_size
    annulus = strategy.sky_annulus

    # Create a list of contrast separations for which to calculate the contrast
    bounds = grid.bounds()
    ncontrast = strategy.ncontrast
    contrasts = np.zeros(ncontrast)
    contrast_separations = np.linspace(0 + aperture, bounds['xmax'] - annulus[1], ncontrast)
    contrast_azimuth = np.radians(strategy.contrast_azimuth)
    contrast_xys = [(separation * np.sin(contrast_azimuth),
                     separation * np.cos(contrast_azimuth)) for separation in contrast_separations]

    # Calculate contrast at each separation
    for i, contrast_xy in enumerate(contrast_xys):
        strategy.target_xy = contrast_xy
        extracted = strategy.extract(my_detector_signal_list, my_detector_noise_list)
        contrasts[i] = extracted['extracted_noise']
    
    extract_unocculted = strategy.extract()

    # What is the flux of the unocculted star.
    # We set the do_contrast attribute to True so the unocculted dither will be used.
    strategy.do_contrast = True
    strategy.on_target = [False, False, True]
    strategy.target_xy = strategy.unocculted_xy
    extract_unocculted = strategy.extract(my_detector_signal_list, my_detector_noise_list)

    # when a source is offset to unocculted_xy, it can be bright enough to cause saturation
    # flags to be raised.  however, since this is an "artifactual" offset, those saturation
    # flags are bogus. the hackish fix is to pop this bogus saturation map off the list
    # and append a new one filled with zeros.
    bogus_sat = my_detector_saturation_list.pop()
    my_detector_saturation_list.append(np.zeros(bogus_sat.shape))

    # Contrast is relative to the unocculted on-axis star.
    contrasts /= extract_unocculted['extracted_flux']
    contrast_curve = [contrast_separations, contrasts]

    # #### END calculation #### #

    # Add the contrast curve and link relevant saturation maps to the extracted_sn dict for passing
    extracted_sn['contrast_curve'] = contrast_curve

    r = Report(input, my_detector_signal_list, my_detector_noise_list, my_detector_saturation_list, extracted_sn, warnings)
    return r
