from __future__ import absolute_import

import sys

class EngineConfiguration(object):
    '''
    A class to consolidate the options for customizing
    calculations made with the Pandiea engine.

    Users aren't expected to interact with this class
    directly. Use the engine.options object instead.

    Could add type checking in the @property.setter
    methods.
    '''
    
    def __init__(self, **kwargs):
        self.default_noise = {'crs': True, 'darkcurrent': True, 'ffnoise': True, 'readnoise': True, 
                              'rn_correlation': True}
        self.default_effects = {'background': True, 'ipc': True, 'saturation': True}
        self.default_params = {'wave_sampling': None, 'on_the_fly_PSFs': False,
                               'on_the_fly_webbpsf_options': {}, 'on_the_fly_webbpsf_opd': None,
                               'on_the_fly_oversample': 3, 'pandeia_fixed_seed': False, 
                               'cache': 'ram', 'noise': self.default_noise, 
                               'effects': self.default_effects, 'verbose': False}
        for item in self.default_params.keys():
            setattr(self, "_"+item, kwargs.get(item, self.default_params[item]))
        self._config = None
        self._saved_options = None
    
    @property
    def current_config(self):
        """
        A space to store a pandeia configuration dictionary for inspection at runtime.
        """
        return self._config
    
    @current_config.setter
    def current_config(self, value):
        """
        Allows you to store a pandeia configuration dictionary for runtime inspection
        """
        self._config = value

    @property
    def verbose(self):
        return self._verbose
    
    @verbose.setter
    def verbose(self, value):
        if isinstance(value, bool):
            self._verbose = value
    
    @property
    def current_options(self):
        """
        Stores all configuration options in a dictionary, and returns that dictionary.
        """
        config_dict = {}
        for item in self.default_params.keys():
            config_dict[item] = getattr(self, item)
        return config_dict
    
    @current_options.setter
    def current_options(self, config):
        """
        If supplied with a dictionary, attempts to set all configuration options from that
        dictionary, with the following precedence:
            - parameter name used by the object
            - current value of the configuration option
        """
        if isinstance(config, dict):
            for item in self.default_params.keys():
                setattr(self, item, config.get(item, getattr(self, item)))
    
    def save_options(self):
        """
        The configuration object has room for a single backed-up option set. This is useful for
        storing a default set before running a calculation, so that you can fiddle with the 
        parameters, run the calculation, and then restore everything to what it was before you
        started fiddling. Note that there is *only* one such built-in save, so if you want to save
        a set of parameters you've created without overwriting that save, you should call the
        current_options parameter and store the resulting dictionary somewhere.
        """
        self._saved_options = self.current_options
    
    def restore_options(self):
        """
        This restores whatever is stored in self._saved_options. If there's nothing there, it will 
        have no effect (rather than crashing your program, so there's that at least). Note that it 
        does *not* delete the configuration stored in self._saved_options.
        """
        self.current_options = self._saved_options
    
    def restore_defaults(self):
        """
        Restore all parameters to their default values.
        """
        for item in self.default_params:
            setattr(self, "_"+item, self.default_params[item])

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
    def on_the_fly_oversample(self):
        '''
        A dictionary of extra options for configuring the PSF 
        calculation ad hoc. Note some options are overridden 
        in get_psf_on_the_fly().
        
        See WebbPSF documentation for more details.
        '''
        return self._on_the_fly_oversample
    
    @on_the_fly_oversample.setter
    def on_the_fly_oversample(self, value):
        if sys.version_info[0] >= 3:
            var_types = (int,)
        else:
            var_types = (int, long)
        if isinstance(value, var_types):
            self._on_the_fly_oversample = value
        
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
    
    @pandeia_noise.setter
    def pandeia_noise(self, value):
        if isinstance(value, dict):
            self._noise = value

    @property
    def pandeia_effects(self):
        '''
        Allows access to the pandeia noise configuration
        '''
        return self._effects
    
    @pandeia_effects.setter
    def pandeia_effects(self, value):
        if isinstance(value, dict):
            self._effects = value

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
