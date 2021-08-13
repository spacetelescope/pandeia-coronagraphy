import os
import json
import numpy as np

import pandeia.engine.instrument_factory as pif

from synphot import SourceSpectrum, SpectralElement
from synphot.models import Empirical1D
import astropy.units as u

#Load configuration files for NIRCam and MIRI to extract details from. 
nircam_config_file = os.path.join(os.environ.get("pandeia_refdata"), 'jwst/nircam/config.json')
miri_config_file = os.path.join(os.environ.get("pandeia_refdata"), 'jwst/miri/config.json')
try:
    with open(nircam_config_file, 'r') as f:
        nircam_config_dict = json.load(f)
except IOError:
    raise IOError("Couldn't locate Pandeia reference files and/or pandeia_data/jwst/nircam/config.json file.")
try:
    with open(miri_config_file, 'r') as f:
        miri_config_dict = json.load(f)
except IOError:
    raise IOError("Couldn't locate Pandeia reference files and/or pandeia_data/jwst/miri/config.json file.")

nircam_filters = [x for x in nircam_config_dict['filters']]
miri_filters = [x for x in miri_config_dict['filters']]

nircam_coro_filters = [x for x in nircam_filters if x != 'wlp4' and (int(x[1:3]) > 17 or (int(x[1]) < 2 and x[-1] == 'n'))]
miri_coro_filters = [x for x in miri_filters if 'c' in x]

nircam_supported_subarrays = [x for x in nircam_config_dict['subarrays'] if not any(y in x for y in ['grism', 'output', 'x', 'tats'])]
miri_supported_subarrays = [x for x in miri_config_dict['subarrays'] if x != 'slitlessprism']

nircam_readout_patterns = [x for x in nircam_config_dict['readout_patterns']]
miri_readout_patterns = [x for x in miri_config_dict['readout_patterns']]

def determine_instrument(filt):
    if filt in nircam_filters:
        instrument = 'nircam'
    elif filt in miri_filters:
        instrument = 'miri'
    
    return instrument

def determine_aperture(filt, nircam_aperture, mode):
    if 'imaging' in mode:
        if filt in nircam_filters:
            if int(filt[1:3]) < 24:
                aperture = 'sw'
            else:
                aperture = 'lw'
        elif filt in miri_filters:
            aperture = 'imager'
    elif mode == 'coronagraphy':
        if filt in nircam_coro_filters:
            if nircam_aperture != 'default':
                aperture = nircam_aperture
            else:
                if int(filt[1:3]) < 17:
                    #Short wavelength
                    aperture = 'mask210r'
                elif filt == 'f277w':
                    #This filter can only be done by the LWB
                    aperture = 'lwb'
                else:
                    #Remaining filters
                    aperture  = 'mask335r'
        elif filt in miri_coro_filters:
            #Each MIRI coronagraphic filter uniquely tied to a aperture
            if '2300' in filt:
                #Must be Lyot aperture
                aperture = 'lyot2300'
            else:
                aperture = 'fqpm' + filt[1:-1]
    
    return aperture

def determine_subarray(filt, mode, nircam_subarray, miri_subarray):
    if filt in nircam_filters:
        if nircam_subarray != 'default':
            subarray = nircam_subarray
        elif mode == 'coronagraphy':
            if int(filt[1:3]) < 24:
                subarray = 'sub640' #Short wavelength channel
            else:
                subarray = 'sub320' #Long wavelength channel
        elif mode == 'imaging':
            subarray = 'sub640' #This is just an arbitrary imaging subarrray
    elif filt in miri_filters:
        if miri_subarray != 'default':
            subarray = miri_subarray
        elif mode == 'coronagraphy':
            subarray = 'mask'+filt[1:-1]
        elif mode == 'imaging':
            subarray = 'brightsky' #Again, just arbitrary imaging subarray
    
    return subarray

def determine_pixel_scale(instrument, filt):
    if instrument == 'miri':
        pixel_scale = 0.11
    elif instrument == 'nircam' and int(filt[1:3]) < 24:
        pixel_scale = 0.0311
    elif instrument == 'nircam' and int(filt[1:3]) > 24:
        pixel_scale = 0.063
    else:
        raise ValueError('Unable to estimate pixel scale from instrument "{}" and filter "{}"'.format(instrument, filt))

    return pixel_scale

def determine_exposure_time(subarray, pattern, groups, integrations):
    if pattern in miri_readout_patterns:
        #Must be MIRI
        nframe =  miri_config_dict['readout_pattern_config'][pattern]['nframe']
        subarray_frame_time = miri_config_dict['subarray_config']['default'][subarray]['tframe'] 
        exposure_time = subarray_frame_time * nframe * groups * integrations 
    elif pattern in nircam_readout_patterns:
        #Must be NIRCam
        subarray_frame_time = nircam_config_dict['subarray_config']['default'][subarray]['tframe'] 
        subarray_tfffr = nircam_config_dict['subarray_config']['default'][subarray]['tfffr']
        nframe = nircam_config_dict['readout_pattern_config'][pattern]['nframe']
        nskip = nircam_config_dict['readout_pattern_config'][pattern]['nskip']
        exposure_time = (subarray_tfffr * integrations) + subarray_frame_time * (integrations + integrations * ((groups - 1) * (nframe + nskip) + nframe))
    else:
        raise ValueError('Provided readout pattern {} not recognised.'.format(pattern))

    return exposure_time

def determine_bar_offset(filt):
    # Returns the bar offset in arcsseconds for the nircam coronagraphs
    all_offsets = nircam_config_dict['bar_offsets']
    try:
        offset = all_offsets[filt.lower()]
    except:
        raise ValueError('Invalid filter selection: {}, could not identify bar offset'.format(filt))

    return offset

def sequence_input_checks(exposures, mode, nircam_aperture, nircam_subarray, miri_subarray, telescope, rolls, nircam_sgd, miri_sgd):
    #Check each exposure has been formatted correctly by the user, and filter/pattern selection is valid.
    for exposure in exposures:
        exposure_error_message = 'Exposure ({}) not understood. Exposures must be of format (*filt*, "optimise", *duration_seconds*) OR (*filt*, *pattern*, *ngroups*, *nints*)'
        if len(exposure) != 3 and len(exposure) != 4:
            raise ValueError(exposure_error_message.format(', '.join(map(str,exposure))))

        if not (isinstance(exposure[0], str) or not isinstance(exposure[1], str)) :
            raise ValueError(exposure_error_message.format(', '.join(map(str,exposure))))

        for value in exposure[2:]:
            if not isinstance(value, (int,float)):
                raise ValueError(exposure_error_message.format(', '.join(map(str,exposure))))

        ####
        #Ensure filter input is correct for each exposure.
        ####
        filt = exposure[0].lower()
        if filt in nircam_filters:
            if mode == 'coronagraphy' and filt not in nircam_coro_filters:
                raise ValueError('Chosen filter "{}" not compatible with NIRCam coronagraphy. Compatible filters are: {}'.format(filt, ', '.join(nircam_coro_filters)))
            if len(exposure) == 4:
                pattern = exposure[1].lower()
                if pattern not in nircam_readout_patterns:
                    raise ValueError('Chosen pattern {} not compatible with NIRCam. Compatible patterns are: {}'.format(pattern, ', '.join(nircam_readout_patterns)))
        elif filt in miri_filters:
            if mode == 'coronagraphy' and filt not in miri_coro_filters:
                raise ValueError('Chosen filter "{}" not compatible with MIRI coronagraphy. Compatible filters are: {}'.format(filt, ', '.join(miri_coro_filters)))
            if len(exposure) == 4:
                pattern = exposure[1].lower()
                if pattern not in miri_readout_patterns:
                    raise ValueError('Chosen pattern {} not compatible with MIRI. Compatible patterns are: {}'.format(pattern, ', '.join(miri_readout_patterns)))
        else:
            raise ValueError('Chosen filter "{}" incompatible with NIRCam and MIRI, options are: NIRCam ({}) and MIRI ({}).'.format(filt, ', '.join(nircam_filters), ', '.join(miri_filters)))
    
    #Other input checks
    #Ensure selected mode is supported
    supported_modes = ['imaging', 'coronagraphy']
    if mode not in supported_modes:
        raise ValueError('Invalid mode selected. Supported modes are: {}'.format(', '.join(supported_modes)))

    #Ensure selected nircam_aperture is supported
    nircam_coro_apertures = ['mask210r', 'mask335r', 'mask430r', 'maskswb', 'masklwb']
    if nircam_aperture != 'default' and nircam_aperture.lower() not in nircam_coro_apertures:
        raise ValueError('Invalid NIRCam aperture selected. Supported apertures are: {}'.format(', '.join(nircam_coro_apertures)))

    #Ensure subarray is supporte
    if nircam_subarray not in nircam_supported_subarrays and nircam_subarray != 'default':
        raise ValueError('Invalid NIRCam subarray selected. Available subarrays are: {}'.format(', '.join(nircam_supported_subarrays)))

    if miri_subarray not in miri_supported_subarrays and miri_subarray != 'default':
        raise ValueError('Invalid MIRI subarray selected. Available subarrays are: {}'.format(', '.join(miri_supported_subarrays)))

    #Ensure telescopes is supported
    if telescope != 'jwst':
        raise ValueError('"jwst" is currently the only supported telescope.')

    #Ensure rolls are all floats or ints, warn if roll is too large
    if rolls != None:
        if (max(rolls) - min(rolls)) > 15:
            print('WARNING: Roll differences of more than 15 degrees are extremely difficult/impossible to schedule.')
        for roll in rolls:
            if not isinstance(roll, (int, float)):
                raise ValueError('Rolls must be specified as an array of floats or integers, or "max" for rolls at 0 and 14 degrees.')

    #Check small grid dithers match with available patterns
    if nircam_sgd != None:
        nircam_dither_patterns = ['3-POINT-BAR', '5-POINT-BAR', '5-POINT-BOX', '5-POINT-DIAMOND', '9-POINT-CIRCLE']
        if not isinstance(nircam_sgd, str) or nircam_sgd not in nircam_dither_patterns:
            raise ValueError('Invalid NIRCam dither pattern selected. Available patterns are: {}'.format(nircam_dither_patterns)) 
    if miri_sgd != None:
        miri_dither_patterns = ['5-POINT-SMALL-GRID', '9-POINT-SMALL-GRID']
        if not isinstance(miri_sgd, str) or miri_sgd not in miri_dither_patterns:
            raise ValueError('Invalid MIRI dither pattern selected. Available patterns are: {}'.format(miri_dither_patterns)) 

def read_bandpass(bandpass):
    bandpass = bandpass.lower()

    try:
        if '2mass' in bandpass or 'wise' in bandpass:
            #Read in PanCAKE provided 2MASS and WISE bandpasses. 
            with open(os.path.join(os.path.dirname(__file__), "resources", "{}.txt".format(bandpass))) as bandpass_file:
                bandpass_data = np.genfromtxt(bandpass_file).transpose()
                bandpass_wave = bandpass_data[0] * 1e4 #Convert from microns to angstrom
                bandpass_throughput = bandpass_data[1]
            #Convert data to a synphot bandpass
            Bandpass = SpectralElement(Empirical1D, points=bandpass_wave, lookup_table=bandpass_throughput)
        elif 'cousins' in bandpass or 'bessel' in bandpass or 'johnson' in bandpass:
            Bandpass = SpectralElement.from_filter(bandpass)
        else:
            # #A JWST filter
            inst = determine_instrument(bandpass)
            if inst == 'nircam':
                if int(bandpass[1:3]) < 24:
                    mode = 'sw_imaging'
                else:
                    mode = 'lw_imaging'
            elif inst == 'miri':
                mode = 'imaging'

            factory_config = {}
            factory_config['instrument'] = {}
            factory_config['instrument']['instrument'] = inst
            factory_config['instrument']['mode'] = mode
            factory_config['instrument']['filter'] = bandpass
            #factory_config['instrument']['aperture'] = mask
            bandpass_wave = np.arange(0.6, 27, 0.001) #Microns for use with pandeia
            InstrumentConfig = pif.InstrumentFactory(config=factory_config)
            bandpass_throughput = InstrumentConfig.get_total_eff(bandpass_wave)

            bandpass_wave *= 1e4 #Convert to angstrom for synphot
            
            Bandpass = SpectralElement(Empirical1D, points=bandpass_wave, lookup_table=bandpass_throughput)
    except:
        raise ValueError('Input normalisation bandpass not recognised. Currently supported filters are: "2mass_j", "2mass_h", "2mass_ks", "wise_w1", "wise_w2", "wise_w3", "wise_w4", "bessel_j", "bessel_h", "bessel_k", "cousins_r", "cousins_i", "johnson_u", "johnson_b", "johnson_v", "johnson_r", "johnson_i", "johnson_j", "johnson_k", {}, {}.'.format(', '.join(nircam_filters), ', '.join(miri_filters)))

    return Bandpass

def read_coronagraph_transmission(mask):
    mask = mask.upper()
    mask_file = os.path.join(os.path.dirname(__file__), "resources", "{}_2DTRANS.txt".format(mask))
    try:
       transmission = np.loadtxt(mask_file)
    except:
        raise ValueError('Input coronagraphic mask "{}" not recognised. Currently supported masks are: "MASKSWB", "MASKLWB", "MASK210R", "MASK335R", "MASK430R", "FQPM1065", "FQPM1140", "FQPM1550"'.format(mask))
        
    return transmission

