from copy import deepcopy
from glob import glob
import json
import multiprocessing as mp
import os
import pkg_resources
import sys
import warnings

if sys.version_info > (3, 2):
    from functools import lru_cache
else:
    from functools32 import lru_cache

import numpy as np

import pandeia
from pandeia.engine.instrument_factory import InstrumentFactory
from pandeia.engine.psf_library import PSFLibrary
pandeia_get_psf = PSFLibrary.get_psf
pandeia_associate_offset_to_source = PSFLibrary.associate_offset_to_source #MOD 1
from pandeia.engine.perform_calculation import perform_calculation as pandeia_calculation
from pandeia.engine.observation import Observation
pandeia_seed = Observation.get_random_seed
from pandeia.engine.astro_spectrum import * 

try:
    import webbpsf
except ImportError:
    pass

from .config import EngineConfiguration
from . import transformations
from . import templates

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
        pandeia.engine.psf_library.PSFLibrary.get_psf = get_psf_cache_wrapper
        pandeia.engine.psf_library.PSFLibrary.associate_offset_to_source = associate_offset_to_source #Added function
    else:
        pandeia.engine.psf_library.PSFLibrary.get_psf = pandeia_get_psf
        pandeia.engine.psf_library.PSFLibrary.associate_offset_to_source = pandeia_associate_offset_to_source #Original pandeia function
    if options.pandeia_fixed_seed:
        pandeia.engine.observation.Observation.get_random_seed = pandeia_seed
    else:
        pandeia.engine.observation.Observation.get_random_seed = random_seed

    calcfile = deepcopy(calcfile)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category = np.VisibleDeprecationWarning) # Suppress float-indexing warnings
        results = pandeia_calculation(calcfile)

    # Reset the fixed seed state set by the pandeia engine
    # to avoid unexpected results elsewhere
    np.random.seed(None) 

    return results

def associate_offset_to_source(self, sources, instrument, aperture_name):
    '''
    Added azimuth information for use with webbpsf. Pandeia currently does not calculate 
    the PA and assumes azimuthal symmetry resulting in incorrect calculations when using 
    the bar coronagraph. 
    '''
    psf_offsets = self.get_offsets(instrument, aperture_name)
    psf_associations = []
    for source in sources:
        # Currently, we only associate radius, not angle.   
        source_offset_radius = np.sqrt(source.position['x_offset']**2. + source.position['y_offset']**2.)
        source_offset_azimuth = 360*(np.pi+np.arctan2(source.position['x_offset'],source.position['y_offset']))/2/np.pi
        psf_associations.append((source_offset_radius,source_offset_azimuth))

    return psf_associations

def get_psf_cache_wrapper(self,*args,**kwargs):
    '''
    An additional layer to allow the use of lru_cache on
    on-the-fly PSFs (which requires hashable inputs,
    and should not include the 'self' argument that
    was added in pandeia 1.1.1...

    '''
    global latest_on_the_fly_PSF


    # Include the on-the-fly override options in the hash key for the lru_cache
    otf_options = tuple(sorted(options.on_the_fly_webbpsf_options.items()) +
            [options.on_the_fly_webbpsf_opd,])

    # this may be needed in get_psf; extract it so we can avoid
    # passing in 'self', which isn't hashable for the cache lookup
    full_aperture = self._psfs[0]['aperture_name']

    if options.verbose:
        print("Getting PSF for: {}, {}, options={}, aperture={}".format( args, kwargs, otf_options, full_aperture))

    tmp = get_psf(*args,**kwargs,otf_options=otf_options,
            full_aperture=full_aperture)
    latest_on_the_fly_PSF = deepcopy(tmp)
    return tmp


@lru_cache(maxsize=cache_maxsize)
def get_psf( wave, instrument, aperture_name, source_offset=(0, 0), otf_options=None,
        full_aperture=None):
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
    image_mask, pupil_mask, fov_pixels, trim_fov_pixels, pix_scl = parse_aperture(aperture_name)
    ins.image_mask = image_mask
    ins.pupil_mask = pupil_mask

    # Apply any extra options if specified by the user:
    for key in options.on_the_fly_webbpsf_options:
        ins.options[key] = options.on_the_fly_webbpsf_options[key]

    if options.on_the_fly_webbpsf_opd is not None:
        ins.pupilopd = options.on_the_fly_webbpsf_opd

    #get offset
    ins.options['source_offset_r'] = source_offset[0]
    ins.options['source_offset_theta'] = source_offset[1]
    ins.options['output_mode'] = 'oversampled'
    ins.options['parity'] = 'odd'

    psf_result = calc_psf_and_center(ins, wave, source_offset[0], source_offset[1], 3,
                            pix_scl, fov_pixels, trim_fov_pixels=trim_fov_pixels)

    pix_scl = psf_result[0].header['PIXELSCL']
    upsamp = psf_result[0].header['OVERSAMP']
    diff_limit = psf_result[0].header['DIFFLMT']
    psf = psf_result[0].data

    psf = {
        'int': psf,
        'wave': wave,
        'pix_scl': pix_scl,
        'diff_limit': diff_limit,
        'upsamp': upsamp,
        'instrument': instrument,
        'aperture_name': aperture_name,
        'source_offset': source_offset
    }

    return psf

def parse_aperture(aperture_name):
    '''
    Return [image mask, pupil mask, fov_pixels, trim_fov_pixels, pixelscale]
    '''
    
    aperture_keys = ['mask210r','mask335r','mask430r','masklwb','maskswb',
                     'fqpm1065','fqpm1140','fqpm1550','lyot2300']
    assert aperture_name in aperture_keys, \
        'Aperture {} not recognized! Must be one of {}'.format(aperture_name, aperture_keys)

    nc = webbpsf.NIRCam()
    miri = webbpsf.MIRI()

    aperture_dict = {
        'mask210r' : ['MASK210R','CIRCLYOT', 101, None, nc._pixelscale_short],
        'mask335r' : ['MASK335R','CIRCLYOT', 101, None, nc._pixelscale_long],
        'mask430r' : ['MASK430R','CIRCLYOT', 101, None, nc._pixelscale_long],
        'masklwb' : ['MASKLWB','WEDGELYOT', 351, 101, nc._pixelscale_long],
        'maskswb' : ['MASKSWB','WEDGELYOT', 351, 101, nc._pixelscale_short],
        'fqpm1065' : ['FQPM1065','MASKFQPM', 81, None, miri.pixelscale],
        'fqpm1140' : ['FQPM1140','MASKFQPM', 81, None, miri.pixelscale],
        'fqpm1550' : ['FQPM1550','MASKFQPM', 81, None, miri.pixelscale],
        'lyot2300' : ['LYOT2300','MASKLYOT', 81, None, miri.pixelscale]
        }
  
    return aperture_dict[aperture_name]

def calc_psf_and_center(ins, wave, offset_r, offset_theta, oversample, pix_scale, fov_pixels, trim_fov_pixels=None):
    '''
    Following the treatment in pandeia_data/dev/make_psf.py to handle
    off-center PSFs for use as a kernel in later convolutions.
    '''

    if offset_r > 0.:
        #roll back to center
        dx = int(np.rint( offset_r * np.sin(np.deg2rad(offset_theta)) / pix_scale ))
        dy = int(np.rint( offset_r * np.cos(np.deg2rad(offset_theta)) / pix_scale ))
        dmax = np.max([np.abs(dx), np.abs(dy)])

        # pandeia forces offset to nearest integer subsampled pixel.
        # At the risk of having subpixel offsets in the recentering,
        # I'm not sure we want to do this in order to capture
        # small-scale spatial variations properly.
        #ins.options['source_offset_r'] = np.sqrt(dx**2 + dy**2) * pix_scale

        psf_result = ins.calc_psf(monochromatic=wave*1e-6, oversample=oversample, fov_pixels=fov_pixels + 2*dmax)

        image = psf_result[0].data
        image = np.roll(image, dx * oversample, axis=1)
        image = np.roll(image, -dy * oversample, axis=0)
        image = image[dmax * oversample:(fov_pixels + dmax) * oversample,
                      dmax * oversample:(fov_pixels + dmax) * oversample]
        #trim if requested
        if trim_fov_pixels is not None:
            trim_amount = int(oversample * (fov_pixels - trim_fov_pixels) / 2)
            image = image[trim_amount:-trim_amount, trim_amount:-trim_amount]
        psf_result[0].data = image
    else:
        psf_result = ins.calc_psf(monochromatic=wave*1e-6, oversample=oversample, fov_pixels=fov_pixels)

    return psf_result

def ConvolvedSceneCubeinit(self, scene, instrument, background=None, psf_library=None, webapp=False):
    '''
    An almost exact copy of pandeia.engine.astro_spectrum.ConvolvedSceneCube.__init__,
    unfortunately reproduced here to circumvent the wave sampling behavior.

    See the nw_maximal variable toward the end of this function. It's now controlled
    by options.wave_sampling defined in this module.
    '''
    self.warnings = {}
    self.scene = scene
    self.psf_library = psf_library
    self.aper_width = instrument.get_aperture_pars()['disp']
    self.aper_height = instrument.get_aperture_pars()['xdisp']
    self.multishutter = instrument.get_aperture_pars()['multishutter']
    nslice_str = instrument.get_aperture_pars()['nslice']

    if nslice_str is not None:
        self.nslice = int(nslice_str)
    else:
        self.nslice = 1

    self.instrument = instrument
    self.background = background

    self.fov_size = self.get_fov_size()

    # Figure out what the relevant wavelength range is, given the instrument mode
    wrange = self.current_instrument.get_wave_range()

    self.source_spectra = []

    # run through the sources and check their wavelength extents. warn if they fall short of the
    # current instrument configuration's range.
    mins = []
    maxes = []
    key = None
    for i, src in enumerate(scene.sources):
        spectrum = AstroSpectrum(src, webapp=webapp)
        self.warnings.update(spectrum.warnings)
        smin = spectrum.wave.min()
        smax = spectrum.wave.max()
        if smin > wrange['wmin']:
            if smin > wrange['wmax']:
                key = "spectrum_missing_red"
                msg = warning_messages[key] % (smin, smax, wrange['wmax'])
                self.warnings["%s_%s" % (key, i)] = msg
            else:
                key = "wavelength_truncated_blue"
                msg = warning_messages[key] % (smin, wrange['wmin'])
                self.warnings["%s_%s" % (key, i)] = msg
        if smax < wrange['wmax']:
            if smax < wrange['wmin']:
                key = "spectrum_missing_blue"
                msg = warning_messages[key] % (smin, smax, wrange['wmin'])
                self.warnings["%s_%s" % (key, i)] = msg
            else:
                key = "wavelength_truncated_red"
                msg = warning_messages[key] % (smax, wrange['wmax'])
                self.warnings["%s_%s" % (key, i)] = msg

        mins.append(smin)
        maxes.append(smax)

    wmin = max([np.array(mins).min(), wrange['wmin']])
    wmax = min([np.array(maxes).max(), wrange['wmax']])

    # make sure we have something within range and error out otherwise
    if wmax < wrange['wmin'] or wmin > wrange['wmax']:
        msg = "No wavelength overlap between source_spectra [%.2f, %.2f] and instrument [%.2f, %.2f]." % (
            np.array(mins).min(),
            np.array(maxes).max(),
            wrange['wmin'],
            wrange['wmax']
        )
        raise RangeError(value=msg)

    # warn if partial overlap between combined wavelength range of all sources and the instrument's wrange
    if wmin != wrange['wmin'] or wmax != wrange['wmax']:
        key = "scene_range_truncated"
        self.warnings[key] = warning_messages[key] % (wmin, wmax, wrange['wmin'], wrange['wmax'])

    """
    Trim spectrum and do the spectral convolution here on a per-spectrum basis.  Most efficient to do it here
    before the wavelength sets are merged.  Also easier and much more efficient than convolving
    an axis of a 3D cube.
    """
    for src in scene.sources:
        spectrum = AstroSpectrum(src, webapp=webapp)
        # we trim here as an optimization so that we only convolve the section we need of a possibly very large spectrum
        spectrum.trim(wrange['wmin'], wrange['wmax'])
        spectrum = instrument.spectrometer_convolve(spectrum)
        self.source_spectra.append(spectrum)

    """
    different spectra will have different sets of wavelengths. the obvious future
    case will be user-supplied spectra, but this is also true for analytic spectra
    that have different emission/absorption lines. go through each of the spectra,
    merge all of the wavelengths sets into one, and then resample each spectrum
    onto the combined wavelength set.
    """
    self.wave = self.source_spectra[0].wave
    for s in self.source_spectra:
        self.wave = merge_wavelengths(self.wave, s.wave)

    projection_type = instrument.projection_type

    """
    For the spectral projections, we could use the pixel sampling. However, this
    may oversample the cube for input spectra with no narrow features. So we first check
    whether the pixel sampling will give us a speed advantage. Otherwise, do not resample to
    an unnecessarily fine wavelength grid.
    """
    if projection_type in ('spec', 'slitless', 'multiorder'):
        wave_pix = instrument.get_wave_pix()
        wave_pix_trim = wave_pix[np.where(np.logical_and(wave_pix >= wrange['wmin'],
                                                         wave_pix <= wrange['wmax']))]
        if wave_pix_trim.size < self.wave.size:
            self.wave = wave_pix_trim

    """
    There is no inherently optimal sampling for imaging modes, but we resample here to
    a reasonable number of wavelength bins if necessary. This helps keep the cube rendering reasonable
    for large input spectra. Note that the spectrum resampling now uses the flux conserving method
    of pysynphot.
    """
    if projection_type == 'image':
        """
        In practice a value of 200 samples within an imaging configuration's wavelength range
        (i.e. filter bandpass) should be more than enough. Note that because we use pysynphot
        to resample, the flux of even narrow lines is conserved.
        """
        if options.wave_sampling is None:
            nw_maximal = 200 #pandeia default
        else:
            nw_maximal = options.wave_sampling
        if self.wave.size > nw_maximal:
            self.wave = np.linspace(wrange['wmin'], wrange['wmax'], nw_maximal)

    self.nw = self.wave.size
    self.total_flux = np.zeros(self.nw)

    for spectrum in self.source_spectra:
        spectrum.resample(self.wave)
        self.total_flux += spectrum.flux

    # also need to resample the background spectrum
    if self.background is not None:
        self.background.resample(self.wave)

    self.grid, self.aperture_list, self.flux_cube_list, self.flux_plus_bg_list = \
        self.create_flux_cube(background=self.background)

    self.dist = self.grid.dist()
    
pandeia.engine.astro_spectrum.ConvolvedSceneCube.__init__ = ConvolvedSceneCubeinit


def _make_dither_weights(self):
    '''
    Hack to circumvent reference subtraction in pandeia,
    which is currently incorrect, as well as turn off
    the additional calculations for the contrast calculation.
    This gives us about a factor of 3 in time savings.
    '''
    self.dither_weights = [1,0,0] #Pandeia: [1,-1,0]
    del self.calc_type
    
pandeia.engine.strategy.Coronagraphy._make_dither_weights = _make_dither_weights
pandeia.engine.strategy.Coronagraphy._create_weight_matrix = pandeia.engine.strategy.ImagingApPhot._create_weight_matrix

def random_seed(self):
    '''
    The pandeia engine sets a fixed seed of 42.
    Circumvent that here.
    '''
    #np.random.seed(None) # Reset the seed if already set
    #return np.random.randint(0, 2**32 - 1) # Find a new one
    return None
