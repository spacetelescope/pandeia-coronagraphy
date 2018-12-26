from __future__ import absolute_import, print_function

"""
This file contains a number of pancake utilities which may be useful in running pancake.
"""

import os

from pandeia.engine.psf_library import PSFLibrary
from pandeia.engine.source import Source

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
