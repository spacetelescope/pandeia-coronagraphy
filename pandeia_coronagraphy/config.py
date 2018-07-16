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
