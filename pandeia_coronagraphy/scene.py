from __future__ import absolute_import

import itertools
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from .transformations import cart_to_polar, rotate
from .engine import perform_calculation

def create_SGD(calcfile,stepsize=20.e-3, pattern_name=None):
    '''
    Create small grid dither pointing set. There are two
    ways to specify dither patterns:

        stepsize : floating point value for a 3x3 grid.

        pattern_name : string name of a pattern corresponding to
                  one of the named dither patterns in APT.

    If you specify pattern_name, then stepsize is ignored.

    See https://jwst-docs-stage.stsci.edu/display/JTI/NIRCam+Small-Grid+Dithers
    for information on the available dither patterns and their names.
    '''
    #loop to set offsets

    if pattern_name is not None:
        pattern_name = pattern_name.upper()
        if pattern_name == "5-POINT-BOX":
            pointings = [(0,       0),
                         (0.015,   0.015),
                         (-0.015,  0.015),
                         (-0.015, -0.015),
                         (0.015,  -0.015)]
        elif pattern_name == "5-POINT-DIAMOND":
            pointings = [(0,      0),
                         (0,      0.02),
                         (0,     -0.02),
                         (+0.02,  0),
                         (-0.02,  0)]
        elif pattern_name == '9-POINT-CIRCLE':
            pointings = [( 0,      0),
                         ( 0,      0.02),
                         (-0.015,  0.015),
                         (-0.02,   0),
                         (-0.015, -0.015),
                         ( 0.000, -0.02),
                         ( 0.015, -0.015),
                         ( 0.020,  0.0),
                         ( 0.015,  0.015)]
        elif pattern_name == "3-POINT-BAR":
            pointings = [(0,    0),
                         (0.0,  0.015),
                         (0.0, -0.015)]
        elif pattern_name == "5-POINT-BAR":
            pointings = [(0,    0),
                         (0.0,  0.020),
                         (0.0,  0.010),
                         (0.0, -0.010),
                         (0.0, -0.020)]
        else:
            raise ValueError("Unknown pattern_name value; check your input matches exactly an allowed SGD pattern in APT.")
    else:
        steps = [-stepsize,0.,stepsize]
        pointings = itertools.product(steps,steps)
    sgds = []

    for i, (sx, sy) in enumerate(pointings):
        curcalc = deepcopy(calcfile)
        if i > 0:
            errx, erry = get_fsm_error()
            offset_x = sx + errx
            offset_y = sy + erry
        else:
            offset_x = sx
            offset_y = sy
        offset_scene(curcalc['scene'], offset_x, offset_y)
        sgds.append(curcalc)
    return sgds

def get_ta_error(error=5.0e-3):
    ''' 5mas 1-sigma/axis error (~7mas radial)
    '''
    return np.random.normal(loc=0.,scale=error,size=2)

def get_fsm_error(error=2.0e-3):
    return np.random.normal(loc=0.,scale=error,size=2)

def rotate_scene(scene,theta,center=[0.,0.]):
    for source in scene:
        newxy = rotate([source['position']['x_offset'],
                        source['position']['y_offset']],
                        theta,
                        center)
        source['position']['x_offset'] = newxy[0]
        source['position']['y_offset'] = newxy[1]

def offset_scene(scene,x,y):
    for source in scene:
        source['position']['x_offset'] += x
        source['position']['y_offset'] += y

def plot_scene(scene,title,newfig=True):
    if newfig:
        plt.figure(figsize=(5,5))
        plt.subplot(111,polar=True)
    for s in scene:
        r, theta = cart_to_polar([s['position']['x_offset'],s['position']['y_offset']])
        plt.plot(np.deg2rad(theta),r,lw=0,marker='o',ms=10,label=s['id'])
    plt.title(title,y=1.1,fontsize=14)
    plt.legend(numpoints=1,loc='best')
