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
    scene_offsets = library.associate_offset_to_source(scene_sources, instrument, aperture)
    return {'scene': scene_offsets, 'reference': reference_offsets}
