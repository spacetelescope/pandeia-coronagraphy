from __future__ import absolute_import, print_function

# Just build an actual subclass of the necessary JWST classes

from copy import deepcopy
from glob import glob
import json
import logging
import multiprocessing as mp
import os
import pkg_resources
import sys
import warnings
import astropy.units as units
import astropy.io.fits as fits
import scipy.integrate as integrate
import webbpsf
from poppy import poppy_core
from functools import wraps

if sys.version_info[0] >= 3:
    from functools import lru_cache
    from io import StringIO
else:
    from functools32 import lru_cache
    from cStringIO import StringIO

import numpy as np

import pandeia.engine
from pandeia.engine import observation
from pandeia.engine import astro_spectrum as astro
from pandeia.engine import background as bg
from pandeia.engine import coords
from pandeia.engine.config import DefaultConfig
from pandeia.engine.report import Report
from pandeia.engine.scene import Scene
from pandeia.engine.calc_utils import build_empty_scene
from pandeia.engine.custom_exceptions import EngineInputError, EngineOutputError, RangeError, DataError
from pandeia.engine.instrument_factory import InstrumentFactory
from pandeia.engine.strategy import StrategyFactory
from pandeia.engine.pandeia_warnings import etc3d_warning_messages as warning_messages
from pandeia.engine import debug_utils

from pandeia.engine.psf_library import PSFLibrary
from pandeia.engine.strategy import Coronagraphy
from pandeia.engine.constants import SPECTRAL_MAX_SAMPLES
default_SPECTRAL_MAX_SAMPLES = SPECTRAL_MAX_SAMPLES
from pandeia.engine.etc3D import CalculationConfig, DetectorSignal
PandeiaDetectorSignal = DetectorSignal

from .config import EngineConfiguration
from . import templates

cache_maxsize = 256     # Number of monochromatic PSFs stored in an LRU cache
                        # Should speed up calculations that involve modifying things
                        # like exposure time and don't actually require calculating new PSFs.


class CoronagraphyPSFLibrary(PSFLibrary, object):
    '''
    Subclass of the Pandeia PSFLibrary class, intended to allow PSFs to be generated on-the-fly
    via webbpsf rather than using cached PSFs
    '''
    def __init__(self, path=None, aperture='all', cache_path=None):
        from .engine import options
        self._options = options
        self._log("debug", "CUSTOM PSF LIBRARY ACTIVATE!")
        if path is None:
            if "pandeia_refdata" in os.environ:
                tel = 'jwst'
                ins = options.current_config['configuration']['instrument']['instrument'].lower()
                path = os.path.join(os.environ['pandeia_refdata'], tel, ins, 'psfs')
        super(CoronagraphyPSFLibrary, self).__init__(path, aperture)
        self.latest_on_the_fly_PSF = None
        self._cache_path = cache_path
        if cache_path is None:
            self._cache_path = os.getcwd()

    def associate_offset_to_source(self, sources, instrument, aperture_name):
        '''
        Added azimuth information for use with webbpsf. Pandeia currently does not calculate 
        the PA and assumes azimuthal symmetry resulting in incorrect calculations when using 
        the bar coronagraph. 
        '''
        psf_offsets = self.get_offsets(instrument, aperture_name)
        psf_associations = []
        for source in sources:
            source_offset_radius = np.sqrt(source.position['x_offset']**2. + source.position['y_offset']**2.)
            source_offset_azimuth = 360*(np.pi+np.arctan2(source.position['x_offset'],source.position['y_offset']))/2/np.pi
            psf_associations.append((source_offset_radius,source_offset_azimuth))

        return psf_associations
    
    def get_pupil_throughput(self, wave, instrument, aperture_name):
        """
        Intended for pandeia 1.2 compatibility.
        """
        if hasattr(super(CoronagraphyPSFLibrary, self), "get_pupil_throughput"):
            return super(CoronagraphyPSFLibrary, self).get_pupil_throughput(wave, instrument, aperture_name)
        ins = CoronagraphyPSFLibrary._get_instrument(instrument, aperture_name)
        return CoronagraphyPSFLibrary._pupil_throughput(ins)
    
    @staticmethod
    @lru_cache(maxsize=cache_maxsize)
    def get_cached_psf( wave, instrument, aperture_name, oversample=None, source_offset=(0, 0), otf_options=None, full_aperture=None):
        from .engine import options
        #Make the instrument and determine the mode
        ins = CoronagraphyPSFLibrary._get_instrument(instrument, aperture_name, source_offset)
        pix_scl = ins.pixelscale
        fov_pixels = CoronagraphyPSFLibrary.fov_pixels[aperture_name]
        trim_fov_pixels = CoronagraphyPSFLibrary.trim_fov_pixels[aperture_name]
        
        psf_result = CoronagraphyPSFLibrary.calc_psf(ins, wave, source_offset, oversample, pix_scl, 
                                                     fov_pixels, trim_fov_pixels=trim_fov_pixels)

        pupil_throughput = CoronagraphyPSFLibrary._pupil_throughput(ins)
        pix_scl = psf_result[0].header['PIXELSCL']
        upsamp = psf_result[0].header['OVERSAMP']
        diff_limit = psf_result[0].header['DIFFLMT']
        psf = psf_result[0].data
        if len(psf) == 0:
            psf = np.ones((1,1))

        psf = {
            'int': psf,
            'wave': wave,
            'pix_scl': pix_scl,
            'diff_limit': diff_limit,
            'upsamp': upsamp,
            'instrument': instrument,
            'aperture_name': aperture_name,
            'source_offset': source_offset,
            'pupil_throughput': pupil_throughput
        }

        return psf

    def get_psf(self, wave, instrument, aperture_name, oversample=None, source_offset=(0, 0), otf_options=None, full_aperture=None):

        cache = self._options.cache
        if oversample is None:
            oversample = self._options.on_the_fly_oversample

        if source_offset[0] > 50.:
            ins = CoronagraphyPSFLibrary._get_instrument(instrument, aperture_name, source_offset)
            diff_limit = ((((wave*units.micron).to(units.meter).value)/6.5)*units.radian).to(units.arcsec).value
            psf = {
                'int': np.ones((1,1)),
                'wave': wave,
                'pix_scl': ins.pixelscale/oversample,
                'diff_limit': diff_limit,
                'upsamp': oversample,
                'instrument': instrument,
                'aperture_name': aperture_name,
                'source_offset': source_offset,
                'pupil_throughput': self._pupil_throughput(ins)
            }
            return psf

        self._log("info", "Getting {} {} {}... with caching {}".format(instrument, aperture_name, wave, cache))
        if cache == 'disk':
            psf_name = 'cached_{:.5f}_{}_{}_{:.3f}_{:.3f}_{}.fits'.format(wave, instrument, aperture_name, source_offset[0], source_offset[1], oversample)
            if self._have_psf(psf_name):
                self._log("info", " Found in cache")
                psf_flux, pix_scl, diff_limit, pupil_throughput = self._get_psf(psf_name)
                psf = {
                    'int': psf_flux,
                    'wave': wave,
                    'pix_scl': pix_scl,
                    'diff_limit': diff_limit,
                    'upsamp': oversample,
                    'instrument': instrument,
                    'aperture_name': aperture_name,
                    'source_offset': source_offset,
                    'pupil_throughput': pupil_throughput
                }
                return psf
        elif cache == 'ram':
            # At this point, splice in the cache wrapper code, since we're testing moving the lru_cache out of the class to see what happens
            # Include the on-the-fly override options in the hash key for the lru_cache
            otf_options = tuple(sorted(self._options.on_the_fly_webbpsf_options.items()) + [self._options.on_the_fly_webbpsf_opd,])

            # this may be needed in get_psf; extract it so we can avoid
            # passing in 'self', which isn't hashable for the cache lookup
            full_aperture = self._psfs[0]['aperture_name']

            tmp = self.get_cached_psf(wave, instrument, aperture_name, oversample, source_offset, otf_options=otf_options, full_aperture=full_aperture)
            self._log("info", " Cache Stats: {}".format(self.get_cached_psf.cache_info()))
            return tmp

        # Either disk cache miss or no caching
        #Make the instrument and determine the mode
        ins = CoronagraphyPSFLibrary._get_instrument(instrument, aperture_name, source_offset)
        pix_scl = ins.pixelscale
        fov_pixels = CoronagraphyPSFLibrary.fov_pixels[aperture_name]
        trim_fov_pixels = CoronagraphyPSFLibrary.trim_fov_pixels[aperture_name]
    
        psf_result = self.calc_psf(ins, wave, source_offset, oversample, pix_scl, fov_pixels, trim_fov_pixels=trim_fov_pixels)

        pupil_throughput = self._pupil_throughput(ins)
        pix_scl = psf_result[0].header['PIXELSCL']
        upsamp = psf_result[0].header['OVERSAMP']
        diff_limit = psf_result[0].header['DIFFLMT']
        psf = psf_result[0].data
        if len(psf) == 0:
            psf = np.ones((1,1))

        psf = {
            'int': psf,
            'wave': wave,
            'pix_scl': pix_scl,
            'diff_limit': diff_limit,
            'upsamp': upsamp,
            'instrument': instrument,
            'aperture_name': aperture_name,
            'source_offset': source_offset,
            'pupil_throughput': pupil_throughput
        }
        
        if cache == 'disk':
            psf_result[0].header['PUPTHR'] = pupil_throughput
            psf_result.writeto(os.path.join(self._cache_path, psf_name))
            self._log("info", " Created and saved to cache.")

        return psf

    def get_pix_scale(self, instrument, aperture_name):
        """
        Get PSF pixel scale for given instrument/aperture.
        
        OVERRIDE Pandeia so as to make sure that the pixel scale comes out correctly.
        """
        aperture_dict = self.parse_aperture(aperture_name)
        upsample = self.get_upsamp(instrument, aperture_name)
        return aperture_dict[4]/upsample

    def _have_psf(self, psf_name):
        '''
        Determine whether a cached PSF exists for a given combination
        '''
        return os.path.exists(os.path.join(self._cache_path, psf_name))

    def _get_psf(self, psf_name):
        '''
        Return a cached PSF exists for a given combination
        '''
        with fits.open(os.path.join(self._cache_path, psf_name)) as inf:
            pix_scl = inf[0].header['PIXELSCL']
            diff_limit = inf[0].header['DIFFLMT']
            pupil_throughput = inf[0].header['PUPTHR']
            psf = inf[0].data
        return psf, pix_scl, diff_limit, pupil_throughput
    
    @staticmethod
    def _pupil_throughput(ins):
        """
        Determines pupil throughput given a webbpsf instrument object
        """
        optsys = ins._getOpticalSystem()
        ote_pupil = optsys[0].amplitude
        coron_pupil = optsys[-2].amplitude
        pupil_throughput = coron_pupil.sum() / ote_pupil.sum()
        return pupil_throughput
    
    @staticmethod
    def _get_instrument(instrument, aperture_name, source_offset=None):
        from .engine import options as pancake_options
        instrument_config = pancake_options.current_config['configuration']['instrument']
        scene_config = pancake_options.current_config['scene']
        ref_config = pancake_options.current_config['strategy']['psf_subtraction_source']
        if source_offset is None:
            offset_x = max([x['position']['x_offset'] for x in scene_config] + [ref_config['position']['x_offset']])
            offset_y = max([x['position']['y_offset'] for x in scene_config] + [ref_config['position']['y_offset']])
            source_offset_radius = np.sqrt(offset_x**2. + offset_y**2.)
            source_offset_azimuth = 360*(np.pi+np.arctan2(offset_x, offset_y))/2/np.pi
            source_offset = [source_offset_radius, source_offset_azimuth]
        if instrument.upper() == 'NIRCAM':
            ins = webbpsf.NIRCam()
            ins.filter = instrument_config['filter']
            if CoronagraphyPSFLibrary.nircam_mode[aperture_name] == 'lw_imaging':
                ins.detector = 'A5'
                ins.pixelscale = ins._pixelscale_long
        elif instrument.upper() == 'MIRI':
            ins = webbpsf.MIRI()
            ins.filter = instrument_config['filter']
        else:
            raise ValueError('Only NIRCam and MIRI are supported instruments!')
        ins.image_mask = CoronagraphyPSFLibrary.image_mask[aperture_name]
        ins.pupil_mask = CoronagraphyPSFLibrary.pupil_mask[aperture_name]
        for key in pancake_options.on_the_fly_webbpsf_options:
            ins.options[key] = pancake_options.on_the_fly_webbpsf_options[key]
        if pancake_options.on_the_fly_webbpsf_opd is not None:
            ins.pupilopd = pancake_options.on_the_fly_webbpsf_opd
        #get offset
        ins.options['source_offset_r'] = source_offset[0]
        ins.options['source_offset_theta'] = source_offset[1]
        ins.options['output_mode'] = 'oversampled'
        ins.options['parity'] = 'odd'
        return ins
    
    @staticmethod
    def parse_aperture(aperture_name):
        '''
        Return [image mask, pupil mask, fov_pixels, trim_fov_pixels, pixelscale]
        '''
    
        aperture_keys = ['mask210r','mask335r','mask430r','masklwb','maskswb','fqpm1065','fqpm1140','fqpm1550','lyot2300']
        assert aperture_name in aperture_keys, 'Aperture {} not recognized! Must be one of {}'.format(aperture_name, aperture_keys)

        nc = webbpsf.NIRCam()
        miri = webbpsf.MIRI()

        aperture_dict = {
            'mask210r' : ['MASK210R','CIRCLYOT', 101, None, nc._pixelscale_short, 'sw_imaging'],
            'mask335r' : ['MASK335R','CIRCLYOT', 101, None, nc._pixelscale_long, 'lw_imaging'],
            'mask430r' : ['MASK430R','CIRCLYOT', 101, None, nc._pixelscale_long, 'lw_imaging'],
            'masklwb' : ['MASKLWB','WEDGELYOT', 351, 101, nc._pixelscale_long, 'lw_imaging'],
            'maskswb' : ['MASKSWB','WEDGELYOT', 351, 101, nc._pixelscale_short, 'sw_imaging'],
            'fqpm1065' : ['FQPM1065','MASKFQPM', 81, None, miri.pixelscale, 'imaging'],
            'fqpm1140' : ['FQPM1140','MASKFQPM', 81, None, miri.pixelscale, 'imaging'],
            'fqpm1550' : ['FQPM1550','MASKFQPM', 81, None, miri.pixelscale, 'imaging'],
            'lyot2300' : ['LYOT2300','MASKLYOT', 81, None, miri.pixelscale, 'imaging']
            }
    
        return aperture_dict[aperture_name]

    @staticmethod
    def calc_psf(ins, wave, offset, oversample, pix_scale, fov_pixels, trim_fov_pixels=None):
        '''
        Following the treatment in pandeia_data/dev/make_psf.py to handle
        off-center PSFs for use as a kernel in later convolutions.
        '''
        # Split out offset
        offset_r, offset_theta = offset
        # Create an optical system model. This is done because, in order to determine the critical angle, we need this model, and it otherwise
        #    wouldn't be generated until the PSF itself is generated. In this case, we want to generate the model early because we want to make
        #    sure that the observation *isn't* over the critical angle *before* generating the PSF
        optsys = ins._getOpticalSystem(fft_oversample=3, detector_oversample=3, fov_arcsec=None, fov_pixels=fov_pixels)
        # determine the spatial frequency which is Nyquist sampled by the input pupil.
        # convert this to units of cycles per meter and make it not a Quantity
        sf = (1./(optsys.planes[0].pixelscale * 2 * units.pixel)).to(1./units.meter).value
        critical_angle_arcsec = wave*1.e-6*sf*poppy_core._RADIANStoARCSEC
        critical_angle_pixels = int(np.floor(0.5 * critical_angle_arcsec / pix_scale))

        if offset_r > 0.:
            #roll back to center
            dx = int(np.rint( offset_r * np.sin(np.deg2rad(offset_theta)) / pix_scale ))
            dy = int(np.rint( offset_r * np.cos(np.deg2rad(offset_theta)) / pix_scale ))
            dmax = np.max([np.abs(dx), np.abs(dy)])

            psf_result = ins.calc_psf(monochromatic=wave*1e-6, oversample=oversample, fov_pixels=min(critical_angle_pixels, fov_pixels + 2*dmax))
        
            image = psf_result[0].data
            image = np.roll(image, dx * oversample, axis=1)
            image = np.roll(image, -dy * oversample, axis=0)
            image = image[dmax * oversample:(fov_pixels + dmax) * oversample, dmax * oversample:(fov_pixels + dmax) * oversample]
            #trim if requested
            if trim_fov_pixels is not None:
                trim_amount = int(oversample * (fov_pixels - trim_fov_pixels) / 2)
                image = image[trim_amount:-trim_amount, trim_amount:-trim_amount]
            psf_result[0].data = image
        else:
            psf_result = ins.calc_psf(monochromatic=wave*1e-6, oversample=oversample, fov_pixels=min(critical_angle_pixels, fov_pixels))

        return psf_result
    
    def _log(self, level, message):
        """
        A bypass for the inability for Pandeia to do some internal python class serialization if the
        class contains a logger
        """
        logger = logging.getLogger(__name__)
        if not len(logger.handlers):
            logger.addHandler(logging.StreamHandler(sys.stderr))
        logger.setLevel(logging.WARNING)
        if self._options.verbose:
            logger.setLevel(logging.DEBUG)
        if not hasattr(logger, level.lower()):
            print("Logger has no function {}".format(level.lower()))
            print("Message is: {}".format(message))
        logging_fn = getattr(logger, level.lower())
        logging_fn(message)
    
    nircam_mode = {
                    'mask210r': 'sw_imaging', 'mask335r': 'lw_imaging', 'mask430r': 'lw_imaging',
                    'masklwb': 'lw_imaging', 'maskswb': 'sw_imaging', 'fqpm1065': 'imaging',
                    'fqpm1140': 'imaging', 'fqpm1550': 'imaging', 'lyot2300': 'imaging'
                  }

    image_mask = {
                    'mask210r': 'MASK210R', 'mask335r': 'MASK335R', 'mask430r': 'MASK430R',
                    'masklwb': 'MASKLWB', 'maskswb': 'MASKSWB', 'fqpm1065': 'FQPM1065',
                    'fqpm1140': 'FQPM1140', 'fqpm1550': 'FQPM1550', 'lyot2300': 'LYOT2300'
                 }
    
    pupil_mask = {
                    'mask210r': 'CIRCLYOT', 'mask335r': 'CIRCLYOT', 'mask430r': 'CIRCLYOT', 
                    'masklwb': 'WEDGELYOT', 'maskswb': 'WEDGELYOT', 'fqpm1065': 'MASKFQPM', 
                    'fqpm1140': 'MASKFQPM', 'fqpm1550': 'MASKFQPM', 'lyot2300': 'MASKLYOT'
                 }
    
    fov_pixels = {
                    'mask210r': 101, 'mask335r': 101, 'mask430r': 101, 'masklwb': 351, 
                    'maskswb': 351, 'fqpm1065': 81, 'fqpm1140': 81, 'fqpm1550': 81, 
                    'lyot2300': 81
                 }
    
    trim_fov_pixels = {
                        'mask210r': None, 'mask335r': None, 'mask430r': None, 'masklwb': 101, 
                        'maskswb': 101, 'fqpm1065': None, 'fqpm1140': None, 'fqpm1550': None, 
                        'lyot2300': None
                      }


class CoronagraphyConvolvedSceneCube(pandeia.engine.astro_spectrum.ConvolvedSceneCube):
    '''
    This class overrides the ConvolvedSceneCube class, and instead of using SPECTRAL_MAX_SAMPLES it
    looks for a wavelength size that should be present in the 'scene' part of the template
    
    background=None, psf_library=None, webapp=False, empty_scene=False
    '''
    def __init__(self, scene, instrument, **kwargs):
        from .engine import options
        self.coronagraphy_options = options
        self._options = options
        self._log("debug", "CORONAGRAPHY SCENE CUBE ACTIVATE!")
        pandeia.engine.astro_spectrum.SPECTRAL_MAX_SAMPLES = self._max_samples
        if 'psf_library' in kwargs and not isinstance(kwargs['psf_library'], CoronagraphyPSFLibrary):
            kwargs['psf_library'] = CoronagraphyPSFLibrary()
        super(CoronagraphyConvolvedSceneCube, self).__init__(scene, instrument, **kwargs)

    @property
    def _max_samples(self):
        '''
        This is intended to replace a constant with a function. Maybe it works?
        '''
        if self.coronagraphy_options.wave_sampling is None:
            return default_SPECTRAL_MAX_SAMPLES
        return self.coronagraphy_options.wave_sampling

    def _log(self, level, message):
        """
        A bypass for the inability for Pandeia to do some internal python class serialization if the
        class contains a logger
        """
        logger = logging.getLogger(__name__)
        if not len(logger.handlers):
            logger.addHandler(logging.StreamHandler(sys.stderr))
        logger.setLevel(logging.WARNING)
        if self._options.verbose:
            logger.setLevel(logging.DEBUG)
        logging_fn = getattr(logger, level)
        logging_fn(message)


class CoronagraphyDetectorSignal(CoronagraphyConvolvedSceneCube):
    '''
    Override the DetectorSignal to avoid odd issues with inheritance. Unfortunately this currently
    means copying the functions entirely (with changes to which class is used)
    
    webapp=False, order=None, empty_scene=False
    '''
    def __init__(self, observation, calc_config=CalculationConfig(), **kwargs):
        # Get calculation configuration
        self.calculation_config = calc_config

        # Link to the passed observation
        self.observation = observation

        # Load the instrument we're using
        self.current_instrument = observation.instrument
        # save order to the DetectorSignal instance, for convenience purposes
        self.order = None
        if 'order' in kwargs:
            self.order = kwargs['order']
            del kwargs['order']
        # and configure the instrument for that order
        self.current_instrument.order = self.order
        
        # Get optional arguments
        webapp = kwargs.get('webapp', False)
        empty_scene = kwargs.get('empty_scene', False)
        
        # Add coronagraphy-specific PSF library for on-the-fly PSF generation
        kwargs['psf_library'] = CoronagraphyPSFLibrary()

        # how are we projecting the signal onto the detector plane?
        self.projection_type = self.current_instrument.projection_type

        # If we're in a dispersed mode, we need to know which axis the signal is dispersed along
        self.dispersion_axis = self.current_instrument.dispersion_axis()

        # Get the detector parameters (read noise, etc.)
        self.det_pars = self.current_instrument.get_detector_pars()

        # Initialize detector mask
        self.det_mask = 1.0

        # Get the background
        if self.calculation_config.effects['background']:
            self.background = bg.Background(self.observation, webapp=webapp)
        else:
            self.background = None
        
        kwargs['background'] = self.background

        # Then initialize the flux and wavelength grid
        CoronagraphyConvolvedSceneCube.__init__(
            self,
            self.observation.scene,
            self.current_instrument,
            **kwargs
        )
        
        self.warnings.update(self.background.warnings)
        # We have to propagate the background through the system transmission
        # to get the background in e-/s/pixel/micron. The background rate is a 1D spectrum.
        self.bg_fp_rate = self.get_bg_fp_rate()

        # Initialize slice lists
        self.rate_list = []
        self.rate_plus_bg_list = []
        self.saturation_list = []
        self.groups_list = []
        self.pixgrid_list = []

        # Loop over all slices and calculate the photon and electron rates through the
        # observatory for each one. Note that many modes (imaging, etc.) will have just
        # a single slice.
        for flux_cube, flux_plus_bg in zip(self.flux_cube_list, self.flux_plus_bg_list):
            # Rates for the slice without the background
            slice_rate = self.all_rates(flux_cube, add_extended_background=False)

            # Rates for the slice with the background added
            slice_rate_plus_bg = self.all_rates(flux_plus_bg, add_extended_background=True)

            # Saturation map for the slice
            slice_saturation = self.get_saturation_mask(rate=slice_rate_plus_bg['fp_pix'])
            exposure_spec = self.current_instrument.exposure_spec
            if hasattr(exposure_spec, 'get_groups_before_sat'):
                slice_group = exposure_spec.get_groups_before_sat(slice_rate_plus_bg['fp_pix'],
                                                                  self.det_pars['fullwell'])
            else:
                slice_group = self._groups_before_sat(slice_rate_plus_bg['fp_pix'], self.det_pars['fullwell'])

            # The grid in the slice
            slice_pixgrid = self.get_pix_grid(slice_rate)

            # Append all slices to the master lists
            self.rate_list.append(slice_rate)
            self.rate_plus_bg_list.append(slice_rate_plus_bg)
            self.saturation_list.append(slice_saturation)
            self.groups_list.append(slice_group)
            self.pixgrid_list.append(slice_pixgrid)

        # Get the mapping of wavelength to pixels on the detector plane. This is grabbed from the
        # first entry in self.rate_list and is currently defined to be the same for all slices.
        self.wave_pix = self.get_wave_pix()

        # This is also grabbed from the first slice as a diagnostic
        self.fp_rate = self.get_fp_rate()

        # Note that the 2D image due to background alone may have spatial structure due to instrumental effects.
        # Therefore it is calculated here.
        self.bg_pix_rate = self.get_bg_pix_rate()

        # Check to see if the background is saturating
        bgsat = self.get_saturation_mask(rate=self.bg_pix_rate)
        if (np.sum(bgsat) > 0) or (np.isnan(np.sum(bgsat))):
            key = "background_saturated"
            self.warnings[key] = warning_messages[key]

        # Reassemble rates of multiple slices on the detector
        self.rate = self.on_detector(self.rate_list)
        self.rate_plus_bg = self.on_detector(self.rate_plus_bg_list)

        exposure_spec = self.current_instrument.exposure_spec
        if hasattr(exposure_spec, 'get_groups_before_sat'):
            self.ngroup_map = exposure_spec.get_groups_before_sat(slice_rate_plus_bg['fp_pix'],
                                                                  self.det_pars['fullwell'])
        else:
            self.ngroup_map = self._groups_before_sat(slice_rate_plus_bg['fp_pix'], self.det_pars['fullwell'])

        if hasattr(exposure_spec, 'get_saturation_fraction'):
            saturation_fraction = exposure_spec.get_saturation_fraction(self.rate_plus_bg, self.det_pars['fullwell'])
        else:
            saturation_fraction = exposure_spec.saturation_time / (self.det_pars['fullwell'] / self.rate_plus_bg)
        self.fraction_saturation = np.max(saturation_fraction)
        
        self.detector_pixels = self.current_instrument.get_detector_pixels(self.wave_pix)

        # Get the read noise correlation matrix and store it as an attribute.
        if self.det_pars['rn_correlation']:
            self.read_noise_correlation_matrix = self.current_instrument.get_readnoise_correlation_matrix(self.rate.shape)
    
    def spectral_detector_transform(self):
        """
        Create engine API format dict section containing properties of wavelength coordinates
        at the detector plane.

        Returns
        -------
        t: dict (engine API compliant keys)
        """
        t = {}
        t['wave_det_refpix'] = 0
        t['wave_det_max'] = self.wave_pix.max()
        t['wave_det_min'] = self.wave_pix.min()

        # there are currently three projection_type's which are basically detector plane types:
        #
        # 'spec' - where the detector plane is purely dispersion vs. spatial
        # 'slitless' - basically a special case of 'spec' with where dispersion and spatial are mixed
        # 'image' - where the detector plane is purely spatial vs. spatial (i.e. no disperser element)
        #
        # 'IFU' mode is of projection_type='spec' because the mapping from detector X pixels to
        # wavelength is the same for each slice.  this projection_type will work for 'MSA' mode as well
        # because we will only handle one aperture at a time.  'slitless' spectroscopy will mix
        # spatial and dispersion information onto the detector X axis.  however, the detector
        # plane is fundamentally spatial vs. wavelength in that case so it's handled the same as
        # projection_type='spec'. creating a spectrum for a specific target will be handled via the
        # extraction strategy.
        if self.projection_type in ('spec', 'slitless', 'multiorder'):
            t['wave_det_size'] = len(self.wave_pix)
            if len(self.wave_pix) > 1:
                # we don't yet have a way of handling non-linear coordinate transforms here. that said,
                # this is mostly right for most of our cases with nirspec prism being the notable exception.
                # this is also only used for plotting purposes while the true actual wave_pix mapping is used
                # internally for all calculations.
                t['wave_det_step'] = (self.wave_pix[-1] - self.wave_pix[0]) / t['wave_det_size']
            else:
                t['wave_det_step'] = 0.0
            t['wave_det_refval'] = self.wave_pix[0]
        elif self.projection_type == "image":
            t['wave_det_step'] = 0.0
            t['wave_det_refval'] = self.wave_pix[0]
            t['wave_det_size'] = 1
        else:
            message = "Unsupported projection_type: %s" % self.projection_type
            raise EngineOutputError(value=message)
        return t

    def wcs_info(self):
        """
        Get detector coordinate transform as a dict of WCS keyword/value pairs.

        Returns
        -------
        header: dict
            WCS header keys defining coordinate transform in the detector plane
        """
        if self.projection_type == 'image':
            # if we're in imaging mode, the detector sampling is the same as the model
            header = self.grid.wcs_info()
        elif self.projection_type in ('spec', 'slitless', 'multiorder'):
            # if we're in a dispersed mode, dispersion can be either along the X or Y axis. the image outputs in
            # the engine Report are rotated so that dispersion will always appear to be along the X axis with
            # wavelength increasing with increasing X (i. e. dispersion angle of 0).  currently, the only other
            # supported dispersion angle is 90 which is what we get when dispersion_axis == 'y'.
            t = self.grid.as_dict()
            t.update(self.spectral_detector_transform())
            header = {
                'ctype1': 'Wavelength',
                'crpix1': 1,
                'crval1': t['wave_det_min'] - 0.5 * t['wave_det_step'],
                'cdelt1': t['wave_det_step'],
                'cunit1': 'um',
                'cname1': 'Wavelength',
                'ctype2': 'Y offset',
                'crpix2': 1,
                'crval2': t['y_min'] - 0.5 * t['y_step'],
                'cdelt2': -t['y_step'],
                'cunit2': 'arcsec',
                'cname2': 'Detector Offset',
            }
            if self.dispersion_axis == 'y':
                header['ctype2'] = 'X offset'
                header['crval2'] = t['x_min'] - 0.5 * t['x_step'],
                header['cdelt2'] = t['x_step']
        else:
            message = "Unsupported projection_type: %s" % self.projection_type
            raise EngineOutputError(value=message)
        return header

    def get_wave_pix(self):
        """
        Return the mapping of wavelengths to pixels on the detector plane
        """
        return self.rate_list[0]['wave_pix']

    def get_fp_rate(self):
        """
        Return scene flux at the focal plane in e-/s/pixel/micron (excludes background)
        """
        return self.rate_list[0]['fp']

    def get_bg_fp_rate(self):
        """
        Calculate background in e-/s/pixel/micron at the focal plane. Also correct for any excess in predicted background
        if there are pupil losses in the PSF. (#2529)
        """
        bg_fp_rate = self.focal_plane_rate(self.ote_rate(self.background.mjy_pix))
        wave_range = self.current_instrument.get_wave_range()
        pupil_thru = self.current_instrument.psf_library.get_pupil_throughput(wave_range['wmin'],
                                                                              self.current_instrument.instrument[
                                                                                  'instrument'],
                                                                              self.current_instrument.instrument[
                                                                                  'aperture'])
        return bg_fp_rate * pupil_thru

    def get_bg_pix_rate(self):
        """
        Calculate the background on the detector in e-/s/pixel
        """
        bg_pix_rate = self.rate_plus_bg_list[0]['fp_pix'] - self.rate_list[0]['fp_pix']
        return bg_pix_rate

    def on_detector(self, rate_list):
        """
        This will take the list of (pixel) rates and use them create a single detector frame. A single
        image will only have one rate in the list, but the IFUs will have n_slices. There may be other examples,
        such as different spectral orders for NIRISS. It is not yet clear how many different flavors there are, so
        this step may get refactored if it gets too complicated. Observing modes that only have one set of rates
        (imaging and single-slit spectroscopy, for instance) will still go through this, but the operation is trivial.
        """
        aperture_sh = rate_list[0]['fp_pix'].shape
        n_apertures = len(rate_list)
        detector_shape = (aperture_sh[0] * n_apertures, aperture_sh[1])
        detector = np.zeros(detector_shape)

        i = 0
        for rate in rate_list:
            detector[i * aperture_sh[0]:(i + 1) * aperture_sh[0], :] = rate['fp_pix']
            i += 1

        return detector

    def get_pix_grid(self, rate):
        """
        Generate the coordinate grid of the detector plane
        """
        if self.projection_type == 'image':
            grid = self.grid
        elif self.projection_type in ('spec', 'slitless', 'multiorder'):
            nw = rate['wave_pix'].shape[0]
            if self.dispersion_axis == 'x':
                # for slitless calculations, the dispersion axis is longer than the spectrum being dispersed
                # because the whole field of view is being dispersed. 'excess' is the size of the FOV
                # and half will be to the left of the blue end of the spectrum and half to the right of the red end.
                # this is used to create the new spatial coordinate transform for the pixel image on the detector.
                excess = rate['fp_pix'].shape[1] - nw
                pix_grid = coords.IrregularGrid(
                    self.grid.col,
                    (np.arange(nw + excess) - (nw + excess) / 2.0) * self.grid.xsamp
                )
            else:
                excess = rate['fp_pix'].shape[0] - nw
                pix_grid = coords.IrregularGrid(
                    (np.arange(nw + excess) - (nw + excess) / 2.0) * self.grid.ysamp,
                    self.grid.row
                )
            return pix_grid
        else:
            raise EngineOutputError(value="Unsupported projection_type: %s" % self.projection_type)
        return grid

    def all_rates(self, flux, add_extended_background=False):
        """
        Calculate rates in e-/s/pixel/micron or e-/s/pixel given a flux cube in mJy

        Parameters
        ----------
        flux: ConvolvedSceneCube instance
            Convolved source flux cube with flux units in mJy
        add_extended_background: bool (default=False)
            Toggle for including extended background not contained within the flux cube

        Returns
        -------
        products: dict
            Dict of products produced by rate calculation.
                'wave_pix' - Mapping of wavelength to detector pixels
                'ote' - Source rate at the telescope aperture
                'fp' - Source rate at the focal plane in e-/s/pixel/micron
                'fp_pix' - Source rate per pixel
                'fp_pix_no_ipc' - Source rate per pixel excluding effects if inter-pixel capacitance
        """
        # The source rate at the telescope aperture
        ote_rate = self.ote_rate(flux)

        # The source rate at the focal plane in interacting photons/s/pixel/micron
        fp_rate = self.focal_plane_rate(ote_rate)

        # the fp_pix_variance is the variance of the per-pixel electron rate and includes the chromatic effects
        # of quantum yield.
        if self.projection_type == 'image':
            # The wavelength-integrated rate in e-/s/pixel, relevant for imagers
            fp_pix_rate, fp_pix_variance = self.image_rate(fp_rate)
            wave_pix = self.wave_eff(fp_rate)

        elif self.projection_type == 'spec':
            # The wavelength-integrated rate in e-/s/pixel, relevant for spectroscopy
            wave_pix, fp_pix_rate, fp_pix_variance = self.spec_rate(fp_rate)

        elif self.projection_type in ('slitless', 'multiorder'):
            # The wavelength-integrated rate in e-/s/pixel, relevant for slitless spectroscopy
            wave_pix, fp_pix_rate, fp_pix_variance = self.slitless_rate(
                fp_rate,
                add_extended_background=add_extended_background
            )

        else:
            raise EngineOutputError(value="Unsupported projection_type: %s" % self.projection_type)

        # Include IPC effects, if available and requested
        if self.det_pars['ipc'] and self.calculation_config.effects['ipc']:
            kernel = self.current_instrument.get_ipc_kernel()
            fp_pix_rate_ipc = self.ipc_convolve(fp_pix_rate, kernel)
        else:
            fp_pix_rate_ipc = fp_pix_rate

        # fp_pix is the final product. Since there is no reason to
        # carry around the ipc label everywhere, we rename it here.
        products = {
            'wave_pix': wave_pix,
            'ote': ote_rate,
            'fp': fp_rate,
            'fp_pix': fp_pix_rate_ipc,
            'fp_pix_no_ipc': fp_pix_rate,  # this is for calculating saturation
            'fp_pix_variance': fp_pix_variance  # this is for calculating the detector noise
        }
        return products

    def ote_rate(self, flux):
        """
        Calculate source rate in e-/s/pixel/micron at the telescope entrance aperture given
        a flux cube in mJy/pixel.
        """
        # spectrum in mJy/pixel, wave in micron, f_lambda in photons/cm^2/s/micron
        f_lambda = 1.5091905 * (flux / self.wave)
        ote_int = self.current_instrument.telescope.get_ote_eff(self.wave)
        coll_area = self.current_instrument.telescope.coll_area
        a_lambda = coll_area * ote_int
        # e-/s/pixel/micron
        ote_rate = f_lambda * a_lambda
        return ote_rate

    def focal_plane_rate(self, rate):
        """
        Takes the output from self.ote_rate() and multiplies it by the components
        of efficiency within the system and returns the source rate at the focal plane in
        e-/s/pixel/micron.
        """
        filter_eff = self.current_instrument.get_filter_eff(self.wave)
        disperser_eff = self.current_instrument.get_disperser_eff(self.wave)
        internal_eff = self.current_instrument.get_internal_eff(self.wave)
        qe = self.current_instrument.get_detector_qe(self.wave)

        fp_rate = rate * filter_eff * disperser_eff * internal_eff * qe
        return fp_rate

    def spec_rate(self, rate):
        '''
        For slitted spectrographs, calculate the detector signal by integrating
        along the dispersion direction of the cube (which is masked by a, by assumption,
        narrow slit). For slitless systems or slits wider than the PSF, the slitless_rate
        method should be used to preserve spatial information within the slit.

        Parameters
        ---------
        rate: numpy.ndarray
            Rate of photons interacting with detector as a function of model wavelength set

        Returns
        -------
        products: 3-element tuple of numpy.ndarrays
            first element - map of pixel to wavelength
            second element - electron rate per pixel
            third element - variance of electron rate per pixel
        '''
        dispersion = self.current_instrument.get_dispersion(self.wave)
        wave_pix = self.current_instrument.get_wave_pix()
        wave_pix_trunc = wave_pix[np.where(np.logical_and(wave_pix >= self.wave.min(),
                                                          wave_pix <= self.wave.max()))]

        # Check that the source spectrum is actually inside the instrumental wavelength
        # coverage.
        if len(wave_pix_trunc) == 0:
            raise RangeError(value='wave and wave_pix do not overlap')

        # Check the dispersion axis to determine which axis to sum and interpolate over
        if self.dispersion_axis == 'x':
            axis = 1
        else:
            axis = 0

        # We can simply sum over the dispersion direction. This is where we lose the spatial information within the aperture.
        spec_rate = np.sum(rate, axis=axis)

        # And then scale to the dispersion function (pixel/micron) to transform
        # from e-/s/micron to e-/s/pixel.
        spec_rate_pix = spec_rate * dispersion

        # but we are still sampled on the internal grid, so we have to interpolate to the pixel grid.
        # use kind='slinear' since it's ~2x more memory efficient than 'linear'. 'slinear' uses different code path to
        # calculate the slopes.
        int_spec_rate = sci_int.interp1d(self.wave, spec_rate_pix, axis=axis, kind='slinear', assume_sorted=True,
                                         copy=False)
        spec_rate_pix_sampled = int_spec_rate(wave_pix_trunc)

        # Handle a detector gap here by constructing a mask. If the current_instrument implements it,
        # it'll be a real mask array.  Otherwise it will simply be 1.0.
        self.det_mask = self.current_instrument.create_gap_mask(wave_pix_trunc)

        # this is the interacting photon rate in the detector with mask applied.
        spec_rate_pix_sampled *= self.det_mask

        # Add effects of non-unity quantum yields. For the spec projection, we assume that the quantum yield does not
        # change over a spectral element. Then we can just multiply the products by the relevant factors.
        q_yield, fano_factor = self.current_instrument.get_quantum_yield(wave_pix_trunc)

        # convert the photon rate to electron rate by multiplying by the quantum yield which is a function of wavelength
        spec_electron_rate_pix = spec_rate_pix_sampled * q_yield

        # to meet IDT expectations, some instruments require a possibly chromatic fudge factor to be applied
        # to the per-pixel electron rate variance.
        var_fudge = self.current_instrument.get_variance_fudge(wave_pix_trunc)

        # the variance in the electron rate, Ve, is also scaled by the quantum yield plus a fano factor which is
        # analytic in the simple 1 or 2 electron case: Ve = (qy + fano) * Re.  since Re is the photon rate
        # scaled by the quantum yield, Re = qy * Rp, we get: Ve = qy * (qy + fano) * Rp
        spec_electron_variance_pix = spec_rate_pix_sampled * q_yield * (q_yield + fano_factor) * var_fudge
        products = wave_pix_trunc, spec_electron_rate_pix, spec_electron_variance_pix

        return products

    def image_rate(self, rate):
        '''
        Calculate the electron rate for imaging modes by integrating along
        the wavelength direction of the cube.

        Parameters
        ---------
        rate: numpy.ndarray
            Rate of photons interacting with detector as a function of model wavelength set

        Returns
        -------
        products: 2-element tuple of numpy.ndarrays
            first element - electron rate per pixel
            second element - variance of electron rate per pixel
        '''
        q_yield, fano_factor = self.current_instrument.get_quantum_yield(self.wave)

        # convert the photon rate to electron rate by multiplying by the quantum yield which is a function of wavelength
        electron_rate_pix = integrate.simps(rate * q_yield, self.wave)

        # to meet IDT expectations, some instruments require a possibly chromatic fudge factor to be applied
        # to the per-pixel electron rate variance.
        var_fudge = self.current_instrument.get_variance_fudge(self.wave)

        # the variance in the electron rate, Ve, is also scaled by the quantum yield plus a fano factor which is
        # analytic in the simple 1 or 2 electron case: Ve = (qy + fano) * Re.  since Re is the photon rate
        # scaled by the quantum yield, Re = qy * Rp, we get: Ve = qy * (qy + fano) * Rp
        electron_variance_pix = integrate.simps(rate * q_yield * (q_yield + fano_factor) * var_fudge, self.wave)

        products = electron_rate_pix, electron_variance_pix

        return products

    def slitless_rate(self, rate, add_extended_background=True):
        '''
        Calculate the detector rates for slitless modes. Here we retain all spatial information and build
        up the detector plane by shifting and coadding the frames from the convolved flux cube. Also need to handle
        and add background that comes from outside the flux cube, but needs to be accounted for.

        Parameters
        ----------
        rate: 3D numpy.ndarray
            Cube containing the flux rate at the focal plane
        add_extended_background: bool (default: True)
            Toggle for including extended background not contained within the flux cube

        Returns
        -------
        products: 2 entry tuple
            wave_pix: 1D numpy.ndarray containing wavelength to pixel mapping on the detector plane
            spec_rate: 2D numpy.ndarray of detector count rates
        '''
        wave_pix = self.current_instrument.get_wave_pix()
        wave_subs = np.where(
            np.logical_and(
                wave_pix >= self.wave.min(),
                wave_pix <= self.wave.max()
            )
        )
        wave_pix_trunc = wave_pix[wave_subs]

        if len(wave_pix_trunc) == 0:
            raise RangeError(value='wave and wave_pix do not overlap')

        dispersion = self.current_instrument.get_dispersion(wave_pix_trunc)
        trace = self.current_instrument.get_trace(wave_pix_trunc)

        q_yield, fano_factor = self.current_instrument.get_quantum_yield(wave_pix_trunc)
        # if we kind='slinear' since it's ~2x more memory efficient than 'linear'. 'slinear' uses different code
        # path to calculate the slopes. However, slinear is *much* slower, so it is a tradeoff. Also lowering the
        # rate type to float32 to conserve memory.
        int_rate_pix = sci_int.interp1d(self.wave, rate.astype(np.float32, casting='same_kind'),
                                        kind='linear', axis=2, assume_sorted=True, copy=False)
        rate_pix = int_rate_pix(wave_pix_trunc)

        # convert the photon rate to electron rate by multiplying by the quantum yield which is a function of wavelength
        electron_rate_pix = rate_pix * q_yield

        # to meet IDT expectations, some instruments require a possibly chromatic fudge factor to be applied
        # to the per-pixel electron rate variance.
        var_fudge = self.current_instrument.get_variance_fudge(wave_pix_trunc)

        # the variance in the electron rate, Ve, is also scaled by the quantum yield plus a fano factor which is
        # analytic in the simple 1 or 2 electron case: Ve = (qy + fano) * Re.  since Re is the photon rate
        # scaled by the quantum yield, Re = qy * Rp, we get: Ve = qy * (qy + fano) * Rp
        electron_variance_pix = rate_pix * q_yield * (q_yield + fano_factor) * var_fudge

        # interpolate the background onto the pixel spacing
        int_bg_fp_rate = sci_int.interp1d(self.wave, self.bg_fp_rate.astype(np.float32, casting='same_kind'),
                                          kind='linear', assume_sorted=True, copy=False)
        bg_fp_rate_pix = int_bg_fp_rate(wave_pix_trunc)

        # calculate electron rate and variance due to background
        bg_electron_rate = bg_fp_rate_pix * q_yield
        bg_electron_variance = bg_fp_rate_pix * q_yield * (q_yield + fano_factor) * var_fudge

        # The first part of this code is meant to add the PSF images from the convolved scene cube along either the x
        # or y axis depending on the dispersion axis, optionally following the path of a spectral trace (currently used
        # only for SOSS mode). The psfs will be added to all locations within the resolution element.
        #
        # Because, in slitless modes, the disperser is dispersing light coming in from everywhere in the pupil plane,
        # every part of the detector should have a contribution from every wavelength of light (from both orders, for
        # SOSS mode). The add_extended_background statement does that - it fills every pixel up to i, and after
        # i+rate_pix.shape[1], with the same background that comes baked into the rate_pix images thanks to the
        # AdvancedPSF functions that create the convolved scene cube.

        if self.empty_scene:
            # if we have an explicitly empty scene, we're doing a background-only order calculation and don't need to
            # even pretend to disperse the spectrum - CombinedSignal will handle padding it to match the interesting
            # order(s) and this way there will be no need to trim.
            spec_shape = (rate_pix.shape[0], rate_pix.shape[1])
            spec_rate = np.zeros(spec_shape)
            spec_variance = np.zeros(spec_shape)
            if add_extended_background:
                for i in np.arange(dispersion.shape[0]):
                    spec_rate += bg_electron_rate[i] * dispersion[i]
                    spec_variance += bg_electron_variance[i] * dispersion[i]
        else:  # if the scene is data
            # dispersion_axis tells us whether we need to sum the planes of the cube horizontally
            # or vertically on the detector plane.
            if self.dispersion_axis == 'x':
                spec_shape = (rate_pix.shape[0], rate_pix.shape[2] + rate_pix.shape[1])
                spec_rate = np.zeros(spec_shape)
                spec_variance = np.zeros(spec_shape)
                for i in np.arange(dispersion.shape[0]):
                    # Background not yet completely added. Make sure there is a trace shift to be done so that we
                    # don't make an expensive call to shift() if we don't have to. Use mode='nearest' to fill in new
                    # pixels with background when image is shifted.
                    if trace[i] != 0.0:
                        spec_rate[:, i:i + rate_pix.shape[1]] += shift(
                            electron_rate_pix[:, :, i],
                            shift=(trace[i], 0),
                            mode='nearest',
                            order=1
                        ) * dispersion[i]
                        spec_variance[:, i:i + rate_pix.shape[1]] += shift(
                            electron_variance_pix[:, :, i],
                            shift=(trace[i], 0),
                            mode='nearest',
                            order=1
                        ) * dispersion[i]
                    else:
                        spec_rate[:, i:i + rate_pix.shape[1]] += electron_rate_pix[:, :, i] * dispersion[i]
                        spec_variance[:, i:i + rate_pix.shape[1]] += electron_variance_pix[:, :, i] * dispersion[i]

                    # Adding background to all other pixels, unless we are asked not to.
                    if add_extended_background:
                        spec_rate[:, :i] += bg_electron_rate[i] * dispersion[i]
                        spec_rate[:, i + rate_pix.shape[1]:] += bg_electron_rate[i] * dispersion[i]
                        spec_variance[:, :i] += bg_electron_variance[i] * dispersion[i]
                        spec_variance[:, i + rate_pix.shape[1]:] += bg_electron_variance[i] * dispersion[i]
            else:  # if the dispersion is on the y axis
                spec_shape = (rate_pix.shape[2] + rate_pix.shape[0], rate_pix.shape[1])
                spec_rate = np.zeros(spec_shape)
                spec_variance = np.zeros(spec_shape)
                for i in np.arange(dispersion.shape[0]):
                    # Background not yet completely added. Make sure there is a trace shift to be done so that we
                    # don't make an expensive call to shift() if we don't have to. Use mode='nearest' to fill in new
                    # pixels with background when image is shifted.
                    if trace[i] != 0.0:
                        spec_rate[i:i + rate_pix.shape[0], :] += shift(
                            electron_rate_pix[:, :, i],
                            shift=(0, trace[i]),
                            mode='nearest',
                            order=1
                        ) * dispersion[i]
                        spec_variance[i:i + rate_pix.shape[0], :] += shift(
                            electron_variance_pix[:, :, i],
                            shift=(0, trace[i]),
                            mode='nearest',
                            order=1
                        ) * dispersion[i]
                    else:
                        spec_rate[i:i + rate_pix.shape[0], :] += electron_rate_pix[:, :, i] * dispersion[i]
                        spec_variance[i:i + rate_pix.shape[0], :] += electron_variance_pix[:, :, i] * dispersion[i]
                    # Adding background to all other pixels, unless we are asked not to.
                    if add_extended_background:
                        spec_rate[:i, :] += bg_electron_rate[i] * dispersion[i]
                        spec_rate[i + rate_pix.shape[0]:, :] += bg_electron_rate[i] * dispersion[i]
                        spec_variance[:i, :] += bg_electron_variance[i] * dispersion[i]
                        spec_variance[i + rate_pix.shape[0]:, :] += bg_electron_variance[i] * dispersion[i]

        # dispersion_axis determines whether wavelength is the first or second axis
        if self.dispersion_axis == 'x' or self.projection_type == 'multiorder':
            products = wave_pix_trunc, spec_rate, spec_variance
        else:
            # if dispersion is along Y, wavelength increases bottom to top, but Y index increases top to bottom.
            # flip the Y axis to account for this.
            products = wave_pix_trunc, np.flipud(spec_rate), np.flipud(spec_variance)

        return products

    def wave_eff(self, rate):
        rate_tot = np.nansum(rate, axis=0)
        a = np.sum(rate_tot * self.wave)
        b = np.sum(rate_tot)
        if b > 0.0:
            wave_eff = a / b
        else:
            wave_eff = self.wave.mean()
        wave_eff_arr = np.array([wave_eff])
        return wave_eff_arr

    def get_projection_type(self):
        return self.projection_type

    def ipc_convolve(self, rate, kernel):
        fp_pix_ipc = sg.fftconvolve(rate, kernel, mode='same')

        debug_utils.debugarrays.store('etc3D', 'ipc_convolve',
                                      {
                                          'rate': rate,
                                          'kernel': kernel,
                                          'fp_pix_ipc': fp_pix_ipc,
                                          'description': 'This is just a short, unnecessary description.'
                                      })

        return fp_pix_ipc

    def get_saturation_mask(self, rate=None):
        """
        Compute a numpy array indicating pixels with full saturation (2), partial saturation (1) and no saturation (0).

        Parameters
        ----------
        rate: None or 2D np.ndarray
            Detector plane rate image used to build saturation map from

        Returns
        -------
        mask: 2D np.ndarray
            Saturation mask image
        """
        if rate is None:
            rate = self.rate_plus_bg

        saturation_mask = np.zeros(rate.shape)

        if self.calculation_config.effects['saturation']:
            fullwell = self.det_pars['fullwell']
            exp_pars = self.current_instrument.exposure_spec
            unsat_ngroups = exp_pars.get_unsaturated_groups(rate, fullwell)
            ngroup = exp_pars.ngroup

            saturation_mask[(unsat_ngroups < ngroup)] = 1
            saturation_mask[(unsat_ngroups < 2)] = 2

        return saturation_mask

    def _groups_before_sat(self, slope, fullwell):
        """
        Fix for Pandeia 1.2/1.3, since exposure spec doesn't have this in 1.2
        """
        exposure_spec = self.current_instrument.exposure_spec
        tfffr = exposure_spec.tfffr
        nframe = exposure_spec.nframe
        tframe = exposure_spec.tframe
        nskip = exposure_spec.nskip
        time_to_saturation = self.det_pars['fullwell'] / slope.clip(1e-10, np.max(slope))
        if exposure_spec.det_type == 'sias':
            groups_before_sat = (time_to_saturation - tfffr) / (nframe * tframe)
        elif exposure_spec.det_type == 'h2rg':
            groups_before_sat = (((time_to_saturation - tfffr) / tframe) - nframe) / (nframe + nskip + 1.)
        else:
            raise ValueError("Unknown detector type {}".format(exposure_spec.det_type))
        slice_group = np.floor(groups_before_sat)
        return slice_group



class SeparateTargetReferenceCoronagraphy(Coronagraphy):
    '''
    This class is intended to override the cronography behaviour of requiring that the reference 
    source be included in the same calculation template as the observation source.
    '''

    def _create_weight_matrix(self, my_detector_signal_list, my_detector_noise_list):
        """
        This private method creates the weight matrix, a_ij, used for the strategy sum. It gets overridden
        in each strategy. In this case, it applies all weight to the first (and only) target. As such, it
        deliberately uses the weight matrix creation from the ImagingApPhot class
        """

        if len(my_detector_signal_list) > 1:
            message = 'This Strategy Configuration is intended for separate Target, Reference, and Unocculted Source observations, so only one signal is supported.'
            raise UnsupportedError(value=message)

        my_detector_signal = my_detector_signal_list[0]

        aperture = self.aperture_size
        annulus = self.sky_annulus

        # pass target_xy to Signal.grid.dist() to offset the target position
        dist = my_detector_signal.grid.dist(xcen=self.target_xy[0], ycen=self.target_xy[1])

        # sky_subs only takes into account whole pixels which is sufficient for the sky estimation
        # region and for the sanity checking we need to do. however, we need to be more exact for the source extraction
        # region. photutils.geometry provides routines to do this either via subsampling or exact geometric
        # calculation. the exact method is slower, but for the sizes of regions we deal with in the ETC it is not noticeable.
        sky_subs = np.where((dist > annulus[0]) & (dist <= annulus[1]))
        n_sky = len(sky_subs[0])

        # generate the source extraction region mask.
        src_region = my_detector_signal.grid.circular_mask(
            aperture,
            xoff=self.target_xy[0],
            yoff=self.target_xy[1],
            use_exact=self.use_exact,
            subsampling=self.subsampling
        )

        # the src_region mask values are the fraction of the pixel subtended by the aperture so
        # in the range 0.0 to 1.0 inclusive.  the effective number of pixels in the aperture is
        # then the sum of this mask.
        n_aper = src_region.sum()

        # do some more sanity checks to make sure the target and background regions are configured as expected
        self._check_circular_aperture_limits(src_region, sky_subs, my_detector_signal.grid, aperture, annulus)

        weight_matrix = np.matrix(src_region)
        if self.background_subtraction:
            weight_matrix[sky_subs] = -1. * n_aper / n_sky

        # The method also returns a list of 'products': subscripts of the weight matrix that is non-zero.
        # This can also be a list if the strategy returns more than one product (such a spectrum over a
        # number of wavelengths).
        product_subscript = weight_matrix.nonzero()

        # The subscripts returned from a matrix contain a redundant dimension. This removes it.
        # Note that this is not how matrix indexing is formally constructed, but it enforces a rule
        # that product subscripts should always be tuples or regular ndarrays.
        product_subscript = (np.array(product_subscript[0]).flatten(), np.array(product_subscript[1]).flatten())
        return weight_matrix, [product_subscript]
    
