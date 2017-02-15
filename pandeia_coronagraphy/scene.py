import pandeia
from pandeia.engine.instrument_factory import InstrumentFactory
from pandeia.engine.perform_calculation import perform_calculation as pandeia_calculation

def _make_dither_weights(self):
    '''
    Hack to circumvent reference subtraction in pandeia,
    which is currently incorrect.

    REMOVE WHEN FIXED
    '''
    self.dither_weights = [1,0,0] #Pandeia: [1,-1,0]

pandeia.engine.strategy.Coronagraphy._make_dither_weights = _make_dither_weights

import json
import itertools
from copy import deepcopy
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from transformations import cart_to_polar, rotate

def load_calculation(filename):
    with open(filename) as f:
        calcfile = json.load(f)
    return calcfile

def save_calculation(calcfile,filename):
    with open(filename,'w+') as f:
        json.dump(calcfile,f,indent=2)

def calculate_batch(calcfiles,nprocesses=None):
    if nprocesses is None:
        nprocesses = mp.cpu_count()
    pool = mp.Pool(processes=nprocesses)
    results = pool.map(perform_calculation,calcfiles)
    pool.close()
    pool.join()
    return results

def perform_calculation(calcfile):
    '''
    Manually decorate pandeia.engine.perform_calculation to circumvent
    pandeia's tendency to modify the calcfile during the calculation.
    '''
    calcfile = deepcopy(calcfile)
    results = pandeia_calculation(calcfile)

    #get fullwell for instrument + detector combo
    #instrument = InstrumentFactory(config=calcfile['configuration'])
    #fullwell = instrument.get_detector_pars()['fullwell']

    #recompute saturated pixels and populate saturation and detector images appropriately
    #image = results['2d']['detector'] * results['information']['exposure_specification']['ramp_exposure_time']
    #saturation = np.zeros_like(image)
    #saturation[image > fullwell] = 1
    #results['2d']['saturation'] = saturation
    #results['2d']['detector'][saturation.astype(bool)] = np.nan

    return results

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