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
    
    @staticmethod
    @lru_cache(maxsize=cache_maxsize)
    def get_cached_psf( wave, instrument, aperture_name, oversample=None, source_offset=(0, 0), otf_options=None, full_aperture=None):
        from .engine import options
        #Make the instrument and determine the mode
        if instrument.upper() == 'NIRCAM':
            ins = webbpsf.NIRCam()
        
            # WebbPSF needs to know the filter to select the optimal 
            # offset for the bar masks. The filter is not passed into
            # get_psf but is stored in the full aperture name in self._psfs
            if aperture_name in ['masklwb', 'maskswb']:
                # Everything after the aperture name is the filter name.
                full_aperture = self._psfs[0]['aperture_name']
                fname = full_aperture[full_aperture.find(aperture_name) + len(aperture_name):]
                ins.filter = fname
            else:
                ins.filter = options.current_config['configuration']['instrument']['filter']
            if wave > 2.5:
                # need to toggle to LW detector.
                ins.detector='A5'
                ins.pixelscale = ins._pixelscale_long
        elif instrument.upper() == 'MIRI':
            ins = webbpsf.MIRI()
            ins.filter = options.current_config['configuration']['instrument']['filter']
        else:
            raise ValueError('Only NIRCam and MIRI are supported instruments!')
        image_mask, pupil_mask, fov_pixels, trim_fov_pixels, pix_scl, mode = CoronagraphyPSFLibrary.parse_aperture(aperture_name)
        if sys.version_info[0] < 3:
            image_mask = "{}".format(image_mask)
            pupil_mask = "{}".format(pupil_mask)
        ins.image_mask = image_mask
        ins.pupil_mask = pupil_mask

        # Apply any extra options if specified by the user:
        from .engine import options
        for key in options.on_the_fly_webbpsf_options:
            if sys.version_info[0] < 3:
                ins.options[key] = "{}".format(options.on_the_fly_webbpsf_options[key])
            else:
                ins.options[key] = options.on_the_fly_webbpsf_options[key]

        if options.on_the_fly_webbpsf_opd is not None:
            if sys.version_info[0] < 3:
                ins.pupilopd = "{}".format(options.on_the_fly_webbpsf_opd)
            else:
                ins.pupilopd = options.on_the_fly_webbpsf_opd

        #get offset
        ins.options['source_offset_r'] = source_offset[0]
        ins.options['source_offset_theta'] = source_offset[1]
        ins.options['output_mode'] = 'oversampled'
        ins.options['parity'] = 'odd'
        
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
        if instrument.upper() == 'NIRCAM':
            ins = webbpsf.NIRCam()
        
            # WebbPSF needs to know the filter to select the optimal
            # offset for the bar masks. The filter is not passed into
            # get_psf but is stored in the full aperture name in self._psfs
            if aperture_name in ['masklwb', 'maskswb']:
                # Everything after the aperture name is the filter name.
                full_aperture = self._psfs[0]['aperture_name']
                fname = full_aperture[full_aperture.find(aperture_name) + len(aperture_name):]
                ins.filter = fname
            if wave > 2.5:
                # need to toggle to LW detector.
                ins.detector='A5'
                ins.pixelscale = ins._pixelscale_long
        elif instrument.upper() == 'MIRI':
            ins = webbpsf.MIRI()
        else:
            raise ValueError('Only NIRCam and MIRI are supported instruments!')
        image_mask, pupil_mask, fov_pixels, trim_fov_pixels, pix_scl = self.parse_aperture(aperture_name)
        ins.image_mask = image_mask
        ins.pupil_mask = pupil_mask

        # Apply any extra options if specified by the user:
        for key in self._options.on_the_fly_webbpsf_options:
            ins.options[key] = self._options.on_the_fly_webbpsf_options[key]

        if self._options.on_the_fly_webbpsf_opd is not None:
            ins.pupilopd = self._options.on_the_fly_webbpsf_opd

        #get offset
        ins.options['source_offset_r'] = source_offset[0]
        ins.options['source_offset_theta'] = source_offset[1]
        ins.options['output_mode'] = 'oversampled'
        ins.options['parity'] = 'odd'
    
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


class CoronagraphyConvolvedSceneCube(pandeia.engine.astro_spectrum.ConvolvedSceneCube):
    '''
    This class overrides the ConvolvedSceneCube class, and instead of using SPECTRAL_MAX_SAMPLES it
    looks for a wavelength size that should be present in the 'scene' part of the template
    '''
    def __init__(self, scene, instrument, background=None, psf_library=None, webapp=False):
        from .engine import options
        self.coronagraphy_options = options
        self._options = options
        self._log("debug", "CORONAGRAPHY SCENE CUBE ACTIVATE!")
        pandeia.engine.astro_spectrum.SPECTRAL_MAX_SAMPLES = self._max_samples
        super(CoronagraphyConvolvedSceneCube, self).__init__(scene, instrument, background, CoronagraphyPSFLibrary(), webapp)

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


class CoronagraphyDetectorSignal(CoronagraphyConvolvedSceneCube, DetectorSignal):
    '''
    Override the DetectorSignal to avoid odd issues with inheritance. Unfortunately this currently
    means copying the functions entirely (with changes to which class is used)
    '''
    def __init__(self, observation, calc_config=CalculationConfig(), webapp=False, order=None, empty_scene=False):
        # Get calculation configuration
        self.calculation_config = calc_config

        # Link to the passed observation
        self.observation = observation

        # Load the instrument we're using
        self.current_instrument = observation.instrument
        # and configure it for the order we wish to use, if applicable
        self.current_instrument.order = order
        # save to the DetectorSignal instance, for convenience purposes
        self.order = order

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

        # Then initialize the flux and wavelength grid
        CoronagraphyConvolvedSceneCube.__init__(
            self,
            self.observation.scene,
            self.current_instrument,
            background=self.background,
            psf_library=CoronagraphyPSFLibrary(),
            webapp=webapp,
            empty_scene=empty_scene
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
            slice_group = self.current_instrument.exposure_spec.get_groups_before_sat(slice_rate_plus_bg['fp_pix'],
                                                                                      self.det_pars['fullwell'])

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
        self.ngroup_map = self.current_instrument.exposure_spec.get_groups_before_sat(self.rate_plus_bg,
                                                                                      self.det_pars['fullwell'])
        self.fraction_saturation = np.max(
            self.current_instrument.exposure_spec.get_saturation_fraction(self.rate_plus_bg,
                                                                          self.det_pars['fullwell']))
        self.detector_pixels = self.current_instrument.get_detector_pixels(self.wave_pix)

        # Get the read noise correlation matrix and store it as an attribute.
        if self.det_pars['rn_correlation']:
            self.read_noise_correlation_matrix = self.current_instrument.get_readnoise_correlation_matrix(
                self.rate.shape)


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
    
