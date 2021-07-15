from __future__ import absolute_import, print_function
"""
This file contains a number of pancake utilities which may be useful in running pancake.
"""
import os
import json
import numpy as np 
from pandeia.engine.psf_library import PSFLibrary
from pandeia.engine.source import Source
import pandeia.engine.sed as psed

from synphot import SourceSpectrum, SpectralElement, Observation
from synphot.models import Empirical1D
import astropy.units as u

from .engine import calculate_target
from .io import read_bandpass

import requests
import requests.exceptions
import re 

import warnings
from astropy.utils.exceptions import AstropyWarning, AstropyUserWarning
warnings.simplefilter(action='ignore', category=AstropyWarning)
warnings.simplefilter(action='ignore', category=AstropyUserWarning) 

import matplotlib.pyplot as plt

def determine_pandeia_offset(config):
    """
    Uses Pandeia's PSF library to determine which PSF offset pandeia would use for a particular
    configuration (for comparison with the offset that pancake would use in on-the-fly PSF
    generation).
    """
    instrument = config['configuration']['instrument']['instrument']
    aperture = config['configuration']['instrument']['aperture']
    scene_sources, reference_sources = [], []
    for source in config['scene']:
        scene_sources.append(Source(config=source))
    if 'psf_subtraction_source' in config['strategy']:
        reference_sources.append(Source(config=config['strategy']['psf_subtraction_source']))
    path = None
    if "pandeia_refdata" in os.environ:
        tel = 'jwst'
        ins = config['configuration']['instrument']['instrument'].lower()
        path = os.path.join(os.environ['pandeia_refdata'], tel, ins, 'psfs')
    library = PSFLibrary(path=path)
    scene_offsets = library.associate_offset_to_source(scene_sources, instrument, aperture)
    reference_offsets = library.associate_offset_to_source(reference_sources, instrument, aperture)
    return {'scene': scene_offsets, 'reference': reference_offsets}

def stellar_spectrum(stellar_type, bandpass, magnitude):
    """
    Create a spectrum dictionary that assumes a Phoenix model with a key found in pandeia, and set
    the magnitude to the provided value in the provided bandpass (in ABMAG)
    """
    spectrum = {'spectrum_parameters': ['normalization', 'sed']}
    spectrum['sed'] = {'sed_type': 'phoenix', 'key': stellar_type}
    if 'bessel' in bandpass or 'johnson' in bandpass:
        spectrum['normalization'] = {'type': 'photsys', 'bandpass': bandpass, 
                                     'norm_flux': magnitude, 'norm_fluxunit': 'abmag'}
    elif 'miri' in bandpass or 'nircam' in bandpass:
        spectrum['normalization'] = {'type': 'jwst', 'bandpass': bandpass, 'norm_flux': magnitude,
                                     'norm_fluxunit': 'abmag'}
    else:
        spectrum['normalization'] = {'type': 'photsys', 'bandpass': bandpass, 
                                     'norm_flux': magnitude, 'norm_fluxunit': 'abmag'}
    
    return spectrum

def pandeia_spectrum(stellar_type):#, norm_val=5, norm_unit='vegamag', norm_bandpass='2mass_ks', **kwargs):
    """
    Generate a spectrum using the Pandeia Phoenix data files. 
    Returns two arrays containing the wavelength and flux values, unormalised. 
    """
    pandeia_spectra = {'config':'phoenix/spectra.json'}
    psed_wav, psed_flux = psed.Phoenix(key=stellar_type, spectra=pandeia_spectra, webapp=True).get_spectrum() #Webapp=True so that only SpT is required
    PandeiaSED = SourceSpectrum(Empirical1D, points=psed_wav, lookup_table=psed_flux) #In format angstrom and flam 
    
    spectrum_wave = PandeiaSED.waveset.to('micron')
    spectrum_flux = PandeiaSED(spectrum_wave, flux_unit='mJy')

    return spectrum_wave, spectrum_flux

def user_spectrum(filename, wave_unit='micron', flux_unit='mJy'):
    '''
    Read in a user spectrum from a specified file. 
    '''
    if not os.path.isfile(filename):
        raise OSError('File "{}" not located, unable to extract spectrum.'.format(filename))

    with open(filename) as f:
        file_spectrum_data = np.genfromtxt(f).transpose()
        file_spectrum_wave = file_spectrum_data[0]
        file_spectrum_flux = file_spectrum_data[1]

    #Attempt to load in source spectrum and convert input units to the default 'angstrom' and 'flam' using <<
    try:
        UserSED = SourceSpectrum(Empirical1D, points=file_spectrum_wave << u.Unit(wave_unit), lookup_table=file_spectrum_flux << u.Unit(flux_unit))
    except:
        raise ValueError('Error converting input wave_unit/flux_unit to the Pandeia flux units via Synphot.')

    #This SED is in units 'angstrom' and 'flam', need to assign to users units and then convert to the Pandeia units of 'micron' and 'mjy'
    spectrum_wave = UserSED.waveset.to('micron')
    spectrum_flux = UserSED(spectrum_wave, flux_unit='mJy')

    return spectrum_wave, spectrum_flux

def normalise_spectrum(input_wave, input_flux, norm_val=5, norm_unit='vegamag', norm_bandpass='2mass_ks'):

    #Get bandpass for normalisation
    NormBandpass = read_bandpass(norm_bandpass)

    #Create SED, ensuring to specify the input units are the Pandeia defaults micron and mJy
    SED = SourceSpectrum(Empirical1D, points=input_wave << u.Unit('micron'), lookup_table=input_flux << u.Unit('mJy'))

    # Normalize to input values, note that the units are retrieved from astropy, but some are explicitly defined in
    # the synphot package such as vegamag, abmag, flam, fnu. By virtue of running the SourceSpectrum/SpectralElement 
    # classes these synphot units are loaded in but if things change in the future, this may break things. 
    VegaSED = SourceSpectrum.from_vega()
    SED = SED.normalize(norm_val*u.Unit(norm_unit), band=NormBandpass, vegaspec=VegaSED)

    #Return to Pandeia units
    spectrum_wave = SED.waveset.to('micron')
    spectrum_flux = SED(spectrum_wave, flux_unit='mJy')

    return spectrum_wave, spectrum_flux

def normalize_spectrum(*args, **kwargs):
    return normalise_spectrum(*args, **kwargs)

def query_simbad(query_string, query_timeout_sec=5.0, default_spt='a0v', verbose=True):
    """
    Query simbad for details on a target object, adapted from the JWST Coronagraphic Visibility Tool
    Will in current state attempt to extract: RA, Dec, Spectral Type and Kmagnitude
    """
    if not isinstance(query_string, str):
        raise TypeError('Name of source must be a string type')
    try:
        response = requests.get('http://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oF?' + query_string, timeout=query_timeout_sec)
    except (requests.exceptions.ConnectionError):
        raise ConnectionError('Could not access Simbad. Most likely because source name not recognised, but also check internet connection/Simbad website.')

    body = response.text
    query_results = {'ra':None, 'dec':None, 'canonical_id':None, 'norm_val':None, 'spt':None, 'norm_bandpass':None, 'norm_unit':None}

    for line in body.split('\n'):
        # RA and DEC
        if line[:2] == '%J' and query_results['ra'] is None:
            match = re.match(r'%J (\d+\.\d+) ([+\-]\d+\.\d+) .+', line)
            if match is None:
                return None
            query_results['ra'], query_results['dec'] = map(float, match.groups())
        # Canonical ID / HD Number
        elif line[:4] == '%I.0' and query_results['canonical_id'] is None:
            match = re.match('%I.0 (.+)', line)
            if match is None:
                return None
            query_results['canonical_id'] = match.groups()[0]
        # Spectral Type
        elif line[:2] == '%S' and query_results['spt'] is None:
            match = re.match(r'%S ([^\s\n]+)', line)
            if match is None:
                return None
            query_results['spt'] = match.groups()[0]
        # K Magnitude
        elif line[:4] == '%M.K' and query_results['norm_val'] is None:
            match = re.match(r'%M.K (.?\d+\.\d+)', line)
            if match is None:
                return None
            query_results['norm_val'] = float(match.groups()[0])
            #Check that the magnitude is from the 2MASS catalogue
            if '2003yCat.2246' in line: 
                query_results['norm_bandpass'] = '2mass_ks'
            elif '2002yCat.2237' in line:
                query_results['norm_bandpass'] = 'johnson_k'
            else:
                if verbose: print('WARNING: Did not recognise K-band magnitude catalogue, approximating with 2MASS Ks')
                query_results['norm_bandpass'] = '2mass_ks'

            #Check that units are in Vegamag, if not check if in AB mag, if not assume Vega. 
            if 'Vega' in line:
                query_results['norm_unit'] = 'vegamag'
            elif 'AB' in line:
                query_results['norm_unit'] = 'abmag'
            else:
                if verbose: print("WARNING: Couldn't determine magnitude system, assuming Vega magnitudes.")
                query_results['norm_unit'] = 'vegamag'

    # Check if there was no spectral type, if not just approximate 
    if query_results['spt'] == None:
        if verbose: print("WARNING: Failed to obtain spectral type from SIMBAD. Using spectral type {} instead.".format(default_spt))
        query_results['spt'] = default_spt

    # Check if any of the other values didn't get assigned...
    if None in query_results.values():
        none_keys = ', '.join([str(key) for key in query_results if query_results[key] == None])
        raise ValueError('Query to Simbad failed for {}, could not obtain: {}'.format(query_string, none_keys))

    return query_results

def convert_spt_to_pandeia(raw_spectral_type):
    '''
    Function to take a spectral type string, either from Simbad or directly from the user, and return an approximation 
    that Pandeia can use. If the spectral type string cannot be understood, then the spectral type will be assumed as A0V
    '''
    raw_spectral_type = raw_spectral_type.lower()
    pandeia_spectral_type = None

    match_failed = False
    failed_spt = 'a0v'
    failure_message = "WARNING: Failed to approximate spectral type '{}' to Pandeia compatible spectral type. Using spectral type {} instead.".format(raw_spectral_type, failed_spt)

    #Read in default spectral types file to check if there is a match to the input spectral type
    pandeia_valid_spectral_types_file = os.path.join(os.environ.get("pandeia_refdata"), "sed/phoenix/spectra.json")
    try:
        with open(pandeia_valid_spectral_types_file, 'r') as f:
            pandeia_valid_spectral_types_dict = json.load(f)
    except IOError:
        raise IOError("Couldn't locate Pandeia reference files and/or pandeia_data/sed/phoenix/spectra.json file.")

    pandeia_valid_spectral_types = sorted(list(pandeia_valid_spectral_types_dict.keys()))
   
    pandeia_spectral_type = raw_spectral_type
    # If there is a match, simply return the raw spectral type
    if pandeia_spectral_type in pandeia_valid_spectral_types:
        return pandeia_spectral_type
    # If there is not a match, need to approximate one and notify the user. 
    else:
        #Check if star is an O/B/A/F/G/K/M star
        compatible_temp_classes = ['o', 'b', 'a', 'f', 'g', 'k', 'm']
        if pandeia_spectral_type[0] not in compatible_temp_classes:
            print(failure_message)
            return failed_spt

        #Check for binary+ objects, only use the primary if so. 
        if '+' in pandeia_spectral_type:
            pandeia_spectral_type = pandeia_spectral_type.split('+')[0]

        #Weirdest objects already removed, strip out simbad 'peculiarities', won't strip out 'v' for variable spectrum 
        #as it clashes with luminosity class. This is maybe fixable but: very small number of objects affected and tricky
        #to implement as it requires case sensitivity. 
        for peculiarity in ['s', 'e', 'n', 'w', 'h', 'p', 'c', 'r', 'u', '?', '(', ')']:
            pandeia_spectral_type = pandeia_spectral_type.replace(peculiarity, '')
        #Also a bunch of 'I' subclasses that can't be used, approximate all of them to just type 'I'
        for isubclass in ['ia-0', 'ia-0/ia', 'ia', 'ia/iab', 'iab', 'iab-b', 'ib', 'ib-ii', '_0-ia', '_0-ia+']:
            pandeia_spectral_type = pandeia_spectral_type.replace(isubclass, 'i')

        #Check if 'm' for metallic lines present, but not if the first character or a character after '/' or '+'
        if 'm' in pandeia_spectral_type and pandeia_spectral_type[0] != 'm':
            m_index = pandeia_spectral_type.rfind('m') #Search for last 'm' in the string
            if not pandeia_spectral_type[m_index-1] in ['/', '+']:
                #Should be just a metallic line signifier
                pandeia_spt_list = list(pandeia_spectral_type)
                pandeia_spt_list[m_index] = ''
                pandeia_spectral_type = ''.join(pandeia_spt_list)

        # Next need to check length of spectra, idea is to get everything a single temperature (O3-M9) and luminosity class (I/II/III/IV/V/VI)
        # before further processing can be done. 
        if len(pandeia_spectral_type) == 1:
            #Must be one of previuosly checked temperature classes, with no subdivision, assume it is type *5V
            pandeia_spectral_type = '{}5v'.format(pandeia_spectral_type[0])
        elif len(pandeia_spectral_type) == 2: 
            #Check the second string value is a number (temperature signifier)
            try:
                float(pandeia_spectral_type[1])
            except:
                print(failure_message)
                return failed_spt
            else:#If it is a number, assume star is spectral type 'V'. 
                pandeia_spectral_type += 'v'
        elif len(pandeia_spectral_type) == 3:
            #Couple of options, most should be simple like 'a3v', but some complicated like 'K-M' that we can still use
            try:
                #If not a number in the middle, must be something weird
                float(pandeia_spectral_type[1])
            except:
                if pandeia_spectral_type[1] == '-' or pandeia_spectral_type[1] == '/':
                    #Temperature class is uncertain with no luminosity classifer, approximate to the latter temperature class *0V
                    pandeia_spectral_type = '{}0v'.format(pandeia_spectral_type[2])
                elif pandeia_spectral_type[1] == ':':
                    #Just don't know what the temperature class subdivision is, approximate to 5
                    pandeia_spectral_type = pandeia_spectral_type.replace(':', '5')
                else:
                    print(failure_message)
                    return failed_spt
        else:
            #Whole host of options, some standard and some more complex. 
            if ':' in pandeia_spectral_type:
                if pandeia_spectral_type[-1] == ':':
                    #Uncertainty in the last quantity, luminosity or temperature class
                    try:
                        float(pandeia_spectral_type[-2])
                    except:
                        #It could be the luminosity class
                        if pandeia_spectral_type[-2] in ['i', 'v']:
                            pandeia_spectral_type = pandeia_spectral_type.replace(':','')
                        else:
                            print(failure_message)
                            return failed_spt
                    else:
                        #It's the temperature class and there is no luminosity class
                        pandeia_spectral_type = pandeia_spectral_type.replace(':','')
            
            if '/' in pandeia_spectral_type:
                # if 'iii/v' in pandeia_spectral_type:
                #     pandeia_spectral_type = pandeia_spectral_type.replace('iii/v', 'v')

                #Must be some uncertainity in either temperature or luminosity class, or both. 
                split_spectral_type = pandeia_spectral_type.split('/')
                pandeia_spt_lumclass = None
                pandeia_spt_tempclass = None

                if len(split_spectral_type) >= 4:
                    # Too many splits, can handle at best 1 split in temperature and 1 split in luminosity
                    print(failure_message)
                    return failed_spt

                for i in range(len(split_spectral_type)-1):
                    try:
                        #Check if the character just before the split is a number
                        float(split_spectral_type[i][-1])
                    except:
                        #If it isn't... Should be an uncertainty in luminosity class, approximate to I, III, or V
                        if split_spectral_type[-1] == 'i' or split_spectral_type[-1] == 'ii':
                            pandeia_spt_lumclass = 'i'
                        elif split_spectral_type[-1] == 'iii' or split_spectral_type[-1] == 'iv':
                            pandeia_spt_lumclass = 'iii'
                        elif split_spectral_type[-1] == 'v' or split_spectral_type[-1] == 'vi':
                            pandeia_spt_lumclass = 'v'
                        else:
                            print(failure_message)
                            return failed_spt
                    else:
                        #If it is... Should be an uncertainty in temperature class, couple more options
                        try: 
                            #Check if the character *after* the split is a number
                            float(split_spectral_type[i+1][0])
                        except:
                            #If it isn't... temperature class is uncertain between the actual divisions, e.g. K9/M0
                            subdiv_one = float(''.join(c for c in split_spectral_type[i] if c.isdigit() or c == '.'))
                            subdiv_two = float(''.join(c for c in split_spectral_type[i+1] if c.isdigit() or c == '.')) + 10
                            average_subdiv = int(round((subdiv_one+subdiv_two)/2))
                            if average_subdiv >= 10:
                                average_subdiv -= 10
                                pandeia_spt_tempclass = '{}{}'.format(split_spectral_type[i+1][0],average_subdiv)
                            else:
                                pandeia_spt_tempclass = '{}{}'.format(split_spectral_type[0][0],average_subdiv)
                        else:
                            #If it is... temperature class is uncertain between subdivisions, e.g. K8/9
                            subdiv_one = float(''.join(c for c in split_spectral_type[i] if c.isdigit() or c == '.'))
                            subdiv_two = float(''.join(c for c in split_spectral_type[i+1] if c.isdigit() or c == '.'))
                            average_subdiv = int(round((subdiv_one+subdiv_two)/2))
                            pandeia_spt_tempclass = '{}{}'.format(split_spectral_type[0][0],average_subdiv)

                #Combine average classes with each other or the base spectral type. 
                if pandeia_spt_tempclass != None and pandeia_spt_lumclass != None:
                    pandeia_spectral_type = pandeia_spt_tempclass + pandeia_spt_lumclass
                elif pandeia_spt_lumclass == None:
                    #Need to get luminosity class from the base spectral type, final character before lumclass should be a number
                    tempclass_final_index = [x.isdigit() for x in pandeia_spectral_type[::-1]].index(True)
                    if tempclass_final_index == 0:
                        #There is no information on the luminosity class, use 'v'
                        pandeia_spt_lumclass = 'v'
                    else:
                        #There is spectral class information.
                        pandeia_spt_lumclass = pandeia_spectral_type[-tempclass_final_index:]
                    pandeia_spectral_type = pandeia_spt_tempclass + pandeia_spt_lumclass
                elif pandeia_spt_tempclass == None:
                    #Need to get the temperature class from the base spectral type
                    subdiv = float(''.join(c for c in split_spectral_type[0] if c.isdigit() or c == '.'))
                    if subdiv == 9.5:
                        #Will be best approximated by '0' of the next temperature class down i.e. K9.5->M0
                        try:
                            div = compatible_temp_classes[compatible_temp_classes.index(pandeia_spectral_type[0])+1]
                            pandeia_spt_tempclass = div + '0'
                            pandeia_spectral_type = pandeia_spt_tempclass + pandeia_spt_lumclass
                        except:
                            print(failure_message)
                            return failed_spt
                    else:
                        #Simply round the temperature class to the nearest even value. 
                        pandeia_spt_tempclass = pandeia_spectral_type[0] + str(int(round(subdiv)))
                        pandeia_spectral_type = pandeia_spt_tempclass + pandeia_spt_lumclass
                else:
                    print(failure_message)
                    return failed_spt
        
        #After that, check the luminosity class and convert to one Pandeia can use if necessary
        if pandeia_spectral_type[-2:] == 'ii' and pandeia_spectral_type[-3] != 'i':
            #Luminosity class II, approximate to III
            pandeia_spectral_type += 'i'
        elif pandeia_spectral_type[-2:] == 'iv' or pandeia_spectral_type[-2:] == 'vi':
            #Luminosity class IV or VI, approximate to V
            pandeia_spectral_type = pandeia_spectral_type[:-2]
            pandeia_spectral_type += 'v'
        elif pandeia_spectral_type[-1] == 'i' or  pandeia_spectral_type[-1] == 'v':
            #Should only catch the remaining I, III and V, luminosity classes - just pass to avoid error
            pass
        else:
            print(failure_message)
            return failed_spt

    if pandeia_spectral_type not in pandeia_valid_spectral_types:
        #Need to convert cleaned spectral type to the nearest one that Pandeia can use. But first, to ease with the
        #adjustment of the strings we split up the spectral type string.
        pandeia_spectral_type = list(pandeia_spectral_type)
        if len(pandeia_spectral_type) == 3 and pandeia_spectral_type[-1] == 'i':
            #These are the 'i' class stars
            if pandeia_spectral_type[0] == 'o':
                # O star conversions
                if pandeia_spectral_type[1] in ['3', '4', '5', '7']:
                    pandeia_spectral_type[1] = '6'
                elif pandeia_spectral_type[1] == '9':
                    pandeia_spectral_type[1] = '8'
                else:
                    print(failure_message)
                    return failed_spt
            elif pandeia_spectral_type[0] == 'm':
                # M star conversions
                if pandeia_spectral_type[1] == '1':
                    pandeia_spectral_type[1] = '0'
                elif pandeia_spectral_type[1] in ['3', '4', '5', '6', '7', '8', '9']:
                    pandeia_spectral_type[1] = '2'
                else:
                    print(failure_message)
                    return failed_spt
            else:
                #BAFGK star convsersions
                if pandeia_spectral_type[1] in ['1', '2']:
                    pandeia_spectral_type[1] = '0'
                elif pandeia_spectral_type[1] in ['3','4', '6', '7']:
                    pandeia_spectral_type[1] = '5'
                elif pandeia_spectral_type[1] in ['8', '9']:
                    pandeia_spectral_type[0] = compatible_temp_classes[compatible_temp_classes.index(pandeia_spectral_type[0])+1]
                    pandeia_spectral_type[1] = '0'
                else:
                    print(failure_message)
                    return failed_spt
        elif len(pandeia_spectral_type) == 5:
            #These are 'iii' class stars
            if pandeia_spectral_type[0] == 'o' or (pandeia_spectral_type[0] == 'b' and pandeia_spectral_type[1] in ['1', '2']):
                pandeia_spectral_type = 'b0iii'
            elif (pandeia_spectral_type[0] == 'b' and pandeia_spectral_type[1] in ['3','4', '6', '7', '8', '9']) or pandeia_spectral_type[0] == 'a':
                pandeia_spectral_type = 'b5iii'
            elif pandeia_spectral_type[0] == 'f' or (pandeia_spectral_type[0] == 'g' and pandeia_spectral_type[1] in ['1', '2']):
                pandeia_spectral_type = 'g0iii'
            elif (pandeia_spectral_type[0] == 'g' or pandeia_spectral_type[0] == 'k') and pandeia_spectral_type[1] in ['3','4','6', '7']:
                pandeia_spectral_type[1] = '5' #Convert to g5iii or k5iii
            elif (pandeia_spectral_type[0] == 'g' and pandeia_spectral_type[1] in ['8', '9']) or (pandeia_spectral_type[0] == 'k' and pandeia_spectral_type[1] in ['1', '2']):
                pandeia_spectral_type = 'k0iii'
            elif (pandeia_spectral_type[0] == 'k' and pandeia_spectral_type[1] in ['8', '9']) or pandeia_spectral_type[0] == 'm':
                pandeia_spectral_type = 'm0iii'
            else:
                print(failure_message)
                return failed_spt
        elif pandeia_spectral_type[-1] == 'v':
            #These are the 'v' class stars, lots of possibilities:
            if pandeia_spectral_type[0] == 'o' and int(pandeia_spectral_type[1])%2 == 0:
                #O star conversions
                pandeia_spectral_type[1] = str(int(pandeia_spectral_type[1]) - 1)
            elif pandeia_spectral_type[0] == 'b':
                #B star conversions
                if pandeia_spectral_type[1] in ['2', '4', '6']:
                    pandeia_spectral_type[1] = str(int(pandeia_spectral_type[1]) - 1)
                elif pandeia_spectral_type[1] in ['7', '9']:
                    pandeia_spectral_type[1] = '8'
                else:
                    print(failure_message)
                    return failed_spt
            elif pandeia_spectral_type[0] == 'a':
                if pandeia_spectral_type[1] in ['2', '4', '6']:
                    pandeia_spectral_type[1] = str(int(pandeia_spectral_type[1]) - 1)
                elif pandeia_spectral_type[1] == '7':
                    pandeia_spectral_type[1] = '5'
                elif pandeia_spectral_type[1] in ['8', '9']:
                    pandeia_spectral_type = 'f0v'
                else:
                    print(failure_message)
                    return failed_spt
            elif pandeia_spectral_type[0] == 'f' or pandeia_spectral_type[0] == 'g':
                if pandeia_spectral_type[1] in ['1', '3', '6', '9']:
                    pandeia_spectral_type[1] = str(int(pandeia_spectral_type[1]) - 1)
                elif pandeia_spectral_type[1] in ['4', '7']:
                    pandeia_spectral_type[1] = str(int(pandeia_spectral_type[1]) + 1)
                else:
                    print(failure_message)
                    return failed_spt
            elif pandeia_spectral_type[0] == 'k':
                if pandeia_spectral_type[1] in ['1', '3', '6', '8']:
                    pandeia_spectral_type[1] = str(int(pandeia_spectral_type[1]) - 1)
                elif pandeia_spectral_type[1] == '4':
                    pandeia_spectral_type[1] = '5'
                elif pandeia_spectral_type[1] == '9':
                    pandeia_spectral_type = 'm0v'
                else:
                    print(failure_message)
                    return failed_spt
            elif pandeia_spectral_type[0] == 'm':
                if pandeia_spectral_type[1] in ['1', '3']:
                    pandeia_spectral_type[1] = str(int(pandeia_spectral_type[1]) - 1)
                elif pandeia_spectral_type[1] in ['4', '6', '7', '8', '9']:
                    pandeia_spectral_type[1] = 'm5v'
                else:
                    print(failure_message)
                    return failed_spt
        pandeia_spectral_type = ''.join(pandeia_spectral_type)

    # Final check to ensure that the approximation worked, if not then set to failed spt. Unless there is a bug this shouldn't get called.     
    if pandeia_spectral_type not in pandeia_valid_spectral_types:
        print(failure_message)
        return failed_spt

    # Check if pandeia spectral type equals user provided, if not notify user. 
    if pandeia_spectral_type != raw_spectral_type:
        approximated_spt_message = "WARNING: Spectral type '{}' not compatible with Pandeia grid, using spectral type '{}' instead."
        print(approximated_spt_message.format(raw_spectral_type,pandeia_spectral_type))
    
    return pandeia_spectral_type

def optimise_readout(obs_dict, t_exp, optimise_margin, min_sat=1e-6, max_sat=1):
    pattern, groups, integrations = None, None, None
    
    t_margin = t_exp * optimise_margin #Convert optimise margin to margin in seconds

    instrument = obs_dict['configuration']['instrument']['instrument']
    subarray = obs_dict['configuration']['detector']['subarray']

    #Need to extract subarray frame time from config files
    config_file = os.path.join(os.environ.get("pandeia_refdata"), 'jwst/{}/config.json'.format(instrument))
    try:
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
    except IOError:
        raise IOError("Couldn't locate Pandeia reference files and/or pandeia_data/jwst/{}/config.json file.".format(instrument))
    subarray_frame_time = config_dict['subarray_config']['default'][subarray]['tframe'] 

    if instrument == 'nircam':
        #NIRCam optimisation tricky due to the variety of readout patterns. First rule out the BRIGHT1, SHALLOW2, MEDIUM2, and DEEP2
        #patterns as their counterparts have more measured frames. 5 patterns left, in this naive optimisation we will try to use the
        #longest possible ramp without saturating. Might be that better contrast can be obtained further out by letting the innermost
        #region saturate, not going to try and approximate this effect.

        used_patterns = ['deep8', 'medium8', 'shallow4', 'bright2', 'rapid'] #Descending order important as longest pattern checked first
        
        obs_dict['configuration']['detector']['ngroup'] = 2   #If it saturates in  groups, tough to observe
        obs_dict['configuration']['detector']['nint'] = 1
        obs_dict['configuration']['detector']['readout_pattern'] = 'rapid'  

        data = calculate_target(obs_dict)
        sat_frac = data['scalar']['fraction_saturation']
        sat_pix = np.count_nonzero(data['2d']['saturation'])

        #Can't work with groups like MIRI, need to define a maximum integration time before saturation
        sat_time = (max_sat/sat_frac)*2 * subarray_frame_time

        if sat_frac > 1:
            #There is saturation, use fastest possible readout pattern and warn user.
            print('WARNING: {:.1f}% saturation detected with the shortest possible readout settings. {} pixels affected.'.format(100*sat_frac, sat_pix))
            pattern, groups, integrations = 'rapid', 2, 1
        else:
            #No saturation, optimise readout parameters
            best_pattern = None
            for pattern in used_patterns:
                #Extract pattern config information from earlier loaded dict
                pattern_config = config_dict['readout_pattern_config'][pattern]
                nframe  = pattern_config['nframe']
                nskip = pattern_config['nskip']

                #Time for a single integration with two groups (minimum possible)
                one_integration_time = subarray_frame_time * ((nframe+nskip)+nframe)

                if one_integration_time > (t_exp+t_margin) or one_integration_time > sat_time:
                    #This pattern is too long to be used, go to next pattern in the list
                    continue
                else:
                    #Pattern will fit into the desired exposure time, now find optimal parameters (if possible)
                    
                    #First need to calculate max number of groups, which is a little tricky. Official formula is
                    #integration time = Tframe × [(Nframes + Nskip) × (Ngroups -1) + Nframes] and we can rearrange
                    #to find the maximum number of groups given the exposure time. 
                    max_groups_exp = int(np.floor((((t_exp + t_margin) - (nframe * subarray_frame_time))/(subarray_frame_time*(nframe+nskip))) + 1))
                    max_groups_sat = int(np.floor((((sat_time) - (nframe * subarray_frame_time))/(subarray_frame_time*(nframe+nskip))) + 1))
                    max_groups = np.min([max_groups_exp, max_groups_sat])

                    if 'deep' in pattern and max_groups > 20:
                        max_groups = 20 #Maximum number of allowed groups for DEEP readout
                    elif max_groups > 10:
                        max_groups = 10 #Maximum number of allowed groups for all other readouts. 

                    #Now follow a similar methodology as with MIRI
                    best_ngroup = None
                    #Start at highest possible number of groups and work our way down until one fits within the margin. 
                    for ngroup in reversed(range(1,max_groups+1)):
                        integration_time = subarray_frame_time * ((nframe + nskip)*(ngroup-1) + nframe)
                        min_int = int(np.ceil( (t_exp-t_margin) / integration_time))
                        max_int = int(np.floor( (t_exp+t_margin) / integration_time))

                        #Range of possible ints for this number of groups, if one or more match find which one is closest 
                        best_nint = None
                        best_diff = np.inf
                        for nint in range(min_int, max_int+1):
                            t = nint*integration_time
                            diff = abs(t_exp - t)
                            if (diff <= abs(t_exp-t_margin)) and diff < best_diff:
                                #Found a pattern that fits within our range
                                best_pattern = pattern
                                best_ngroup = ngroup
                                best_nint = nint
                                best_diff = diff

                        if best_nint != None:
                            #Have found optimised readout parameters for this pattern within the user defined range, exit optimisation loop. 
                            break
                        else:
                            #Need to keep searching
                            pass

                if best_pattern != None:
                    # Optimised pattern and readout parameters have been found, break out of pattern loop
                    break
                else:
                    #Need to move to the next pattern down
                    pass

            if best_ngroup == None or best_nint == None or best_pattern == None:
                #Optimisation has failed
                raise RuntimeError('Unable to identify optimal readout parameters. Increase "optimise_margin" or manually define readout.')

            #Otherwise, the optimisation was succesful 
            pattern, groups, integrations = best_pattern, best_ngroup, best_nint

    elif instrument == 'miri':
        #MIRI optimisation is relatively straightforward, always use FAST readout pattern, so we just need to find the largest number of 
        #groups that a) doesn't saturate the detector, and b) is within th margin around the submitted time. 

        obs_dict['configuration']['detector']['ngroup'] = 2   #If it saturates in 5 groups, tough to observe
        obs_dict['configuration']['detector']['nint'] = 1
        obs_dict['configuration']['detector']['readout_pattern'] = 'fast'  

        data = calculate_target(obs_dict)
        sat_frac = data['scalar']['fraction_saturation']
        sat_pix = np.count_nonzero(data['2d']['saturation'])
 
        if sat_frac > 1:
            #There is saturation, use fastest possible readout pattern and warn user.
            print('WARNING: {:.1f}% saturation detected with the shortest possible readout settings. {} pixels affected.'.format(100*sat_frac, sat_pix))
            pattern, groups, integrations = 'fast', 2, 1
        else:
            #No saturation, optimise readout parameters. 
            max_groups = int(np.floor((max_sat/sat_frac)*2))
            cosmic_ray_groups = int(np.floor(300/subarray_frame_time))
            if max_groups > cosmic_ray_groups: 
                #Above balanced cosmic ray limit of 300s, force maximum down to the highest number of groups
                max_groups = cosmic_ray_groups

            best_ngroup = None
            #Start at highest possible number of groups and work our way down until one fits within the margin. 
            for ngroup in reversed(range(1,max_groups+1)):
                min_int = int(np.ceil( (t_exp-t_margin) / (subarray_frame_time*ngroup)))
                max_int = int(np.floor( (t_exp+t_margin) / (subarray_frame_time*ngroup)))
                
                #Range of possible ints for this number of groups, if one or more match find which one is closest 
                best_nint = None
                best_diff = np.inf
                for nint in range(min_int, max_int+1):
                    t = subarray_frame_time*ngroup*nint
                    diff = abs(t_exp - t)
                    if (diff <= abs(t_exp-t_margin)) and diff < best_diff:
                        #Found a pattern that fits within our range
                        best_ngroup = ngroup
                        best_nint = nint
                        best_diff = diff

                if best_nint != None:
                    #Have found an optimised readout pattern within the user defined range, exit optimisation loop. 
                    break
                else:
                    #Need to keep searching
                    pass 

            if best_ngroup == None or best_nint == None:
                #Optimisation has failed
                raise RuntimeError('Unable to identify optimal readout parameters. Increase "optimise_margin" or manually define readout.')

            #Otherwise, the optimisation was succesful 
            pattern, groups, integrations = 'fast', best_ngroup, best_nint
    else:
        raise ValueError('Instrument {} is not supported, please select NIRCam or MIRI'.format(instrument))

    return pattern, groups, integrations

def optimize_readout(*args, **kwargs):
    return optimise_readout(*args, **kwargs)

def compute_magnitude(spectrum_wave, spectrum_flux, filt, wave_unit='micron', flux_unit='mJy'):

    Bandpass = read_bandpass(filt)
    SED = SourceSpectrum(Empirical1D, points=spectrum_wave << u.Unit(wave_unit), lookup_table=spectrum_flux << u.Unit(flux_unit))
    VegaSED = SourceSpectrum.from_vega()

    Obs = Observation(SED, Bandpass, binset=Bandpass.waveset)
    magnitude = Obs.effstim(flux_unit='vegamag', vegaspec=VegaSED).value

    return magnitude

def equatorial_to_ecliptic(ra, dec, form=None, verbose=False):
    ecl = 23.43 * (np.pi / 180)

    if form == 'degrees':
        ra *= (np.pi / 180)
        dec *= (np.pi / 180)
    else:
        if verbose:
            print('Assuming angles are in radians')

    beta = np.arcsin(np.sin(dec)*np.cos(ecl) - np.cos(dec)*np.sin(ecl)*np.sin(ra))
    lamb = np.arccos(np.cos(ra)*np.cos(dec) / np.cos(beta)) 

    if form == 'degrees':
        beta *= 180/np.pi
        lamb *=  180/np.pi

    return lamb, beta