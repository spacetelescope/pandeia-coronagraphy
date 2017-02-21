import json
import itertools
from copy import deepcopy
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from transformations import cart_to_polar, rotate
from engine import perform_calculation

def save_to_fits(array,filename):
    hdu = fits.PrimaryHDU(array)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filename)

def create_SGD(calcfile,stepsize=20.e-3):
    '''
    '''
    #loop to set offsets
    steps = [-stepsize,0.,stepsize]
    sgds = []
    for sx,sy in itertools.product(steps,steps):
        curcalc = deepcopy(calcfile)
        errx, erry = get_fsm_error()
        curcalc['scene'][0]['position']['x_offset'] = sx + errx
        curcalc['scene'][0]['position']['y_offset'] = sy + erry
        sgds.append(curcalc)
    return sgds

def get_ta_error(error=5.0e-3):
    ''' 7mas 1-sigma/axis error
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