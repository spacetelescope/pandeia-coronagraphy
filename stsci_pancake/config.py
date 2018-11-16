from __future__ import absolute_import

class EngineConfiguration(object):
    '''
    A class to consolidate the options for customizing
    calculations made with the Pandiea engine.

    Users aren't expected to interact with this class
    directly. Use the engine.options object instead.

    Could add type checking in the @property.setter
    methods.
    '''
    
    def __init__(self):
        self._wave_sampling = None
        self._on_the_fly_PSFs = False
        self._on_the_fly_webbpsf_options = {}
        self._on_the_fly_webbpsf_opd = None
        self._pandeia_fixed_seed = False
        self._cache = 'ram'
        self._noise = {'crs': True, 'darkcurrent': True, 'ffnoise': True, 'readnoise': True, 'rn_correlation': True}
        self._effects = {'background': True, 'ipc': True, 'saturation': True}
        self.verbose=False
        self._config = None
    
    @property
    def current_config(self):
        return self._config
    
    @current_config.setter
    def current_config(self, config):
        self._config = config

    @staticmethod
    def pandeia_instrument_config(aperture):
        aperture_dict = {
            'mask210r' : ['CIRCLYOT', 'nircam_sw'],
            'mask335r' : ['CIRCLYOT', 'nircam_lw'],
            'mask430r' : ['CIRCLYOT', 'nircam_lw'],
            'masklwb'  : ['WEDGELYOT', 'nircam_lw'],
            'maskswb'  : ['WEDGELYOT', 'nircam_sw'],
            'fqpm1065' : ['MASKFQPM', 'miri'],
            'fqpm1140' : ['MASKFQPM', 'miri'],
            'fqpm1550' : ['MASKFQPM', 'miri'],
            'lyot2300' : ['MASKLYOT', 'miri']
            }
        return aperture_dict[aperture]
    
    @property
    def cache(self):
        '''
        Caching can currently be done one of three ways:
            - LRU RAM cache ('ram')
            - On-disk cache ('disk')
            - No caching ('none')
        '''
        return self._cache
    
    @cache.setter
    def cache(self, value):
        if value in ['none', 'disk', 'ram']:
            self._cache = value
    
    @property
    def noise(self):
        '''
        Sets noise parameters. The default is for everything to be turned on.
        '''
        return self._noise
    
    @noise.setter
    def noise(self, value):
        self._noise = value
    
    @property
    def effects(self):
        '''
        Sets pandeia effects (background, saturation, IPC). The default is for everything to be turned on.
        '''
        return self._effects
    
    @effects.setter
    def effects(self, value):
        self._effects = value
    
    @property
    def wave_sampling(self):
        '''
        Pandeia defaults to ~200 wavelength samples for imaging
        This is slow and unlikely to significantly improve the
        accuracy of coronagraphic performance predictions.
        Setting wave_sampling to 10-20 should be sufficient,
        and translates to a time savings of about 10.
        Leaving it at None sets it to Pandeia's default.
        '''
        return self._wave_sampling
    
    @wave_sampling.setter
    def wave_sampling(self, value):
        self._wave_sampling = value
        
    @property
    def on_the_fly_PSFs(self):
        '''
        Avoid Pandeia's precomputed PSFs and recompute in WebbPSF as needed?
        '''
        return self._on_the_fly_PSFs
    
    @on_the_fly_PSFs.setter
    def on_the_fly_PSFs(self, value):
        self._on_the_fly_PSFs = value
        
    @property
    def on_the_fly_webbpsf_options(self):
        '''
        A dictionary of extra options for configuring the PSF 
        calculation ad hoc. Note some options are overridden 
        in get_psf_on_the_fly().
        
        See WebbPSF documentation for more details.
        '''
        return self._on_the_fly_webbpsf_options
    
    @on_the_fly_webbpsf_options.setter
    def on_the_fly_webbpsf_options(self, value):
        self._on_the_fly_webbpsf_options = value
        
    @property
    def on_the_fly_webbpsf_opd(self):
        '''
        Allow overriding the default OPD selection when 
        computing PSFs on the fly.
        
        See the WebbPSF documentation.
        '''
        return self._on_the_fly_webbpsf_opd
    
    @on_the_fly_webbpsf_opd.setter
    def on_the_fly_webbpsf_opd(self, value):
        self._on_the_fly_webbpsf_opd = value
        
    @property
    def pandeia_fixed_seed(self):
        '''
        By default, the pandeia engine uses a fixed seed.
        This has undesirable results for many coronagraphy
        applications. Pandeia-Coronagraphy disables this
        by default. Restore the fixed seed by toggling
        this to True.
        '''
        return self._pandeia_fixed_seed
    
    @pandeia_fixed_seed.setter
    def pandeia_fixed_seed(self, value):
        self._pandeia_fixed_seed = value
    
    @property
    def pandeia_noise(self):
        '''
        Allows access to the pandeia noise configuration
        '''
        return self._noise

    @property
    def pandeia_effects(self):
        '''
        Allows access to the pandeia noise configuration
        '''
        return self._effects

    def set_crs(self, value):
        '''
        Allows for cosmic rays to be turned on or off in the simulation
        '''
        self._noise['crs'] = value

    def set_darkcurrent(self, value):
        '''
        Allows for dark current to be turned on or off in the simulation
        '''
        self._noise['darkcurrent'] = value

    def set_ffnoise(self, value):
        '''
        Allows for flatfield noise to be turned on or off in the simulation
        '''
        self._noise['ffnoise'] = value

    def set_readnoise(self, value):
        '''
        Allows for readnoise to be turned on or off in the simulation
        '''
        self._noise['readnoise'] = value

    def set_rn_correlation(self, value):
        '''
        Allows for readnoise correlation to be turned on or off in the simulation
        '''
        self._noise['rn_correlation'] = value

    def set_background(self, value):
        '''
        Allows for background counts to be turned on or off in the simulation
        '''
        self._effects['background'] = value

    def set_ipc(self, value):
        '''
        Allows for inter-pixel capacitance to be turned on or off in the simulation
        '''
        self._effects['ipc'] = value

    def set_saturation(self, value):
        '''
        Allows for detector saturation to be turned on or off in the simulation
        '''
        self._effects['saturation'] = value
