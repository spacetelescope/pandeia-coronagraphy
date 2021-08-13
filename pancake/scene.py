from __future__ import absolute_import, print_function

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from astropy.io import fits

from .transformations import polar_to_cart, cart_to_polar, rotate
from .engine import perform_calculation
from .utilities import query_simbad, convert_spt_to_pandeia, user_spectrum, pandeia_spectrum, normalise_spectrum, compute_magnitude
from pandeia.engine.calc_utils import build_default_calc

'''
Scene class to mirror the construction of a typical Pandeia 'Scene', reinventing the wheel a little bit, but means that
users don't need to import pandeia in their main code and that aspects of data input can be streamlined. 
'''
class Scene():
    __NEXT_ID = 1
    def __init__(self, name=None, **kwargs):
        #Load a default pandeia scene to assign properties to. NIRCam/Coroangraphy doesn't matter here, just need an empty scene dict
        self.pandeia_scene = build_default_calc('jwst', 'nircam', 'coronagraphy')['scene']
        self.pandeia_scene[0]['assigned'] = False #No source has been assigned to this 'default' scene yet 
        self.source_list = []

        if name == None:
            self.scene_name = 'Scene{}'.format(Scene.__NEXT_ID)
            Scene.__NEXT_ID += 1
        else:
            self.scene_name = name

    def add_source(self, name, kind='simbad', r=0.0, theta=0.0, verbose=True, **kwargs):
        if verbose: print('{} // Adding Source: {}'.format(self.scene_name, name))
        raw_id = len(self.source_list)
        
        if '_' in name or ':' in name:
            raise ValueError('Underscores "_" and colons ":" cannot be included in Scene names for file saving purposes')

        self.source_list.append(name) 
        #Check if the scene dictionary needs to be extended to add another source. 
        if raw_id > 0: self.pandeia_scene.append(deepcopy(self.pandeia_scene[0]))

        #Load in source
        working_source = self.pandeia_scene[raw_id]
        working_source['id'] = raw_id+1 #Each source must have an allocated source ID starting at 1. 
        working_source['pancake_parameters'] = {}
        working_source['pancake_parameters']['name'] = name
        
        #Apply source offset
        #NOTE: polar_to_cart returns a normal conversion, in our case things are flipped as we are working from the y-axis. 
        yoff, xoff = polar_to_cart(r, theta)
        working_source['position']['x_offset'] = -xoff #Negative as we are working N->E counterclockwise
        working_source['position']['y_offset'] = yoff

        # Use 'kind' of source input to read in different properties 
        if kind == 'simbad':
            #Attempt to query data for the input 'name' string from simbad
            query_results = query_simbad(name, verbose=verbose)

            approx_spt = convert_spt_to_pandeia(query_results['spt'])
            working_source['pancake_parameters']['spt'] = approx_spt
            for qresult in ['ra', 'dec', 'norm_bandpass', 'norm_val', 'norm_unit']:
                working_source['pancake_parameters'][qresult] = query_results[qresult]

            #Generate spectrum
            raw_spectrum_wave, raw_spectrum_flux = pandeia_spectrum(working_source['pancake_parameters']['spt'])
            spectrum_wave, spectrum_flux = normalise_spectrum(raw_spectrum_wave, raw_spectrum_flux, norm_val=working_source['pancake_parameters']['norm_val'], norm_unit=working_source['pancake_parameters']['norm_unit'], norm_bandpass=working_source['pancake_parameters']['norm_bandpass'])
        elif kind == 'grid':
            #User provides the spectral type, normalisation bandpass, and normalisation mag for source so a spectrum can be retrieved from a grid
            for ginput in ['spt', 'norm_val', 'norm_unit', 'norm_bandpass']:
                #Check variable has been provided
                try: 
                    ginput_val = kwargs.get(ginput)
                except:
                    raise NameError("Please provide the spectral type ('spt'), normalisation flux ('norm_val'), normalisation flux unit ('norm_unit'), and normalisation bandpass ('norm_bandpass') of the source")

                #Check inputs are of the correct variable types.
                if ginput!='norm_val' and not isinstance(ginput_val, str):
                    raise TypeError("{} input must be of string type.".format(ginput))
                elif ginput=='norm_val' and not isinstance(ginput_val, (int,float)):
                    raise TypeError("{} input must be of int or float type.".format(ginput))
                
                # Assign input values to the scene. 
                if ginput == 'spt':
                    #Find the best approximation spectral type that Pandeia understands (could match what user provides).
                    approx_spt = convert_spt_to_pandeia(kwargs.get(ginput))
                    working_source['pancake_parameters']['spt'] = approx_spt
                else:
                    working_source['pancake_parameters'][ginput] = kwargs.get(ginput)

            #Generate spectrum
            raw_spectrum_wave, raw_spectrum_flux = pandeia_spectrum(working_source['pancake_parameters']['spt'])
            spectrum_wave, spectrum_flux = normalise_spectrum(raw_spectrum_wave, raw_spectrum_flux, norm_val=working_source['pancake_parameters']['norm_val'], norm_unit=working_source['pancake_parameters']['norm_unit'], norm_bandpass=working_source['pancake_parameters']['norm_bandpass'])
        elif kind == 'file':
            #User provides a file location for a spectrum which can then be read in and converted to microns and mJy
            spectrum_wave, spectrum_flux = user_spectrum(kwargs.get('filename'), wave_unit=kwargs.get('wave_unit'), flux_unit=kwargs.get('flux_unit'))
        else:
            raise ValueError("Source generation kind not recognised, please use 'simbad', 'grid', or 'file'")

        #We are taking the normalisation away from Pandeia so that other filters can be added like 2MASS/WISE.
        working_source['spectrum']['normalization']['type'] = 'none' 
        #Assign spectrum of source to the Pandeia scene.  
        working_source['spectrum']['sed']['sed_type'] = 'input'
        working_source['spectrum']['sed']['spectrum'] = [spectrum_wave, spectrum_flux]

    #Renormalise a source
    def renormalise_source(self, source, norm_val=5, norm_unit='vegamag', norm_bandpass='2mass_ks'):
        try:
            raw_id = self.source_list.index(source)
        except:
            raise ValueError('Source {} has not been allocated to this scene. Currently allocated sources are: {}'.format(source, ', '.join(self.source_list)))

        working_source = self.pandeia_scene[raw_id]

        spectrum_wave, spectrum_flux = working_source['spectrum']['sed']['spectrum']
        renorm_spec_wave, renorm_spec_flux = normalise_spectrum(spectrum_wave, spectrum_flux, norm_val=norm_val, norm_unit=norm_unit, norm_bandpass=norm_bandpass)

        working_source['spectrum']['sed']['spectrum'] = [renorm_spec_wave, renorm_spec_flux]

    def source_magnitude(self, source, filt):
        try:
            raw_id = self.source_list.index(source)
        except:
            raise ValueError('Source {} has not been allocated to this scene. Currently allocated sources are: {}'.format(source, ', '.join(self.source_list)))

        spectrum_wave, spectrum_flux = source['spectrum']['sed']['spectrum']

        magnitude = compute_magnitude(spectrum_wave, spectrum_flux, bandpass)

        return magnitude

    #Offset scene in x, y space in arcseconds
    def offset_scene(self,x,y):
        for source in self.pandeia_scene:
            source['position']['x_offset'] += x
            source['position']['y_offset'] += y

    #Rotate scene given an angle in degrees
    def rotate_scene(self, theta, center=[0.,0.], direction='counter_clockwise'):
        if direction == 'counter_clockwise': 
            #Subtract from 360 to convert to a counter clockwise rotation
            theta = 360-theta
        elif direction != 'clockwise':
            raise ValueError('Invalid direction: {}, options are "clockwise" or "counter_clockwise"')
        for source in self.pandeia_scene:
            newxy = rotate([source['position']['x_offset'],source['position']['y_offset']],theta,center)
            source['position']['x_offset'] = newxy[0]
            source['position']['y_offset'] = newxy[1]
    
    def plot_source_spectra(self, sources='all', title='', newfig=True):
        if newfig:
            plt.figure(figsize=(8,5))
            ax = plt.subplot(111)
        for s in self.pandeia_scene:
            if s['pancake_parameters']['name'] in sources or sources == 'all':
                ax.plot(s['spectrum']['sed']['spectrum'][0], s['spectrum']['sed']['spectrum'][1], label=s['pancake_parameters']['name'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.5,30)
        ax.set_ylim(1e-3,None)
        ax.set_title(title,y=1.1,fontsize=14)
        ax.tick_params(which='both', direction='in', labelsize=12, axis='both', top=True, right=True)
        ax.xaxis.set_ticklabels([], minor=True)
        ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g'))
        ax.xaxis.set_minor_locator(tck.FixedLocator([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]))
        ax.xaxis.set_major_locator(tck.FixedLocator([0.6, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 28]))
        ax.set_xlabel('Wavelength ($\mu$m)',fontsize=14)
        ax.set_ylabel('Spectral Flux Density (mJy)',fontsize=14)
        ax.legend(numpoints=1,loc='best')
        plt.show()

    def plot_scene(self, title='', newfig=True):
        if newfig:
            plt.figure(figsize=(5,5))
            ax = plt.subplot(111,projection='polar')
        for s in self.pandeia_scene:
            r, theta = cart_to_polar([s['position']['x_offset'],s['position']['y_offset']])
            theta -= 90 # As we use the y-axis as theta=0, not x
            ax.plot(np.deg2rad(theta),r,lw=0,marker='o',ms=10,label=s['pancake_parameters']['name'])
        ax.set_rmin(0) #Centre the scene at 0,0
        ax.set_title(title,y=1.1,fontsize=14)
        ax.legend(numpoints=1, loc='best', framealpha=1)
        ax.set_theta_offset(np.pi/2)
        plt.show()


def create_SGD(ta_error=False, fsm_error='default', stepsize=20.e-3, pattern_name=None, sim_num=0):
    '''
    Create small grid dither pointing set. There are two
    ways to specify dither patterns:
    
        ta_error : add TA error to each point in the SGD?

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
        elif pattern_name == "SINGLE-POINT":
            pointings = [(0, 0)]
        elif pattern_name == "5-POINT-SMALL-GRID":
            pointings = [( 0,      0),
                         (-0.010,  0.010),
                         ( 0.010,  0.010),
                         ( 0.010, -0.010),
                         (-0.010, -0.010)]
        elif pattern_name == "9-POINT-SMALL-GRID":
            pointings = [( 0,      0),
                         (-0.010,  0.0),
                         (-0.010,  0.010),
                         ( 0.0,    0.010),
                         ( 0.010,  0.010),
                         ( 0.010,  0.0),
                         ( 0.010, -0.010),
                         ( 0.0,   -0.010),
                         (-0.010, -0.010)]
        else:
            raise ValueError("Unknown pattern_name value; check your input matches exactly an allowed SGD pattern in APT.")
    else:
        steps = [-stepsize,0.,stepsize]
        pointings = itertools.product(steps,steps)
    
    if ta_error=='saved':
        # Use a ta_error from a saved list of draws from a 5mas normal distribution
        saved_ta_x = [-0.005097995256803303, 0.0074080606271851816, -0.007066083176404194, -0.008734646144095127, 0.003742012593729784, 0.0016481375982420419, 0.007931135860387031, 0.0071022706754521195, \
        0.0038422111741224065, -0.00174682207232712, 0.0025717599837055778, 0.0039963574194768255, -0.007207685391798244, -0.0008712770453152541, -0.007546448101693021, 0.004193712269134684, 0.006792077664885723, \
        -4.1410580551813435e-05, -0.001105059253226592, -0.0028727892355570497, 0.008350041478090153, -0.0016977651558595232, -0.007223508100938771, 0.0009164276712610287, 0.002160302740883549, 0.0005949333745946268, \
        -0.0074192526395526134, -0.004951812109036682, -0.004263492939164255, 0.0018153226294126093, 0.0017650414801441278, 0.006774046974691265, -0.0028855194506944132, -0.00010423813585424634, -0.0018273541841140007, \
         -0.010191382189425599, -0.0016823540703577173, -0.0015905980543867355, 0.00033661287019594614, 0.002225787950327535, -0.003968516085501142, -0.003030719750018415, -0.006085307725738674, 0.0008357937183166421, \
         -0.0028506557694151914, -0.0036112320467240827, -0.00557657849783731, -0.006046199965288763, -0.0036878304620914367, -0.004276586637038271]
        saved_ta_y = [-0.0032554274327427878, -0.006438053189153058, 0.0029514992455072254, 0.016365340343214817, 0.006511952977945722, 0.0041969355600369, 0.0018407842985093015, -0.005409769649826246, \
        -0.0006634703662271433, -0.0023781070486209237, -0.00333044905289127, -0.003994796563967052, 0.013046231995292428, 0.0052184036105878484, -0.0009957482170979974, 0.005063472027930356, -0.002946069354062212, \
        -0.0011392745210500818, -0.003536792509007751, -0.0019789870108685935, 0.0051994827401879475, 0.004640760132829544, 0.0014599963533062693, -0.0017201060158097414, -0.00015996588734501168, 0.0018546406036391563, \
        0.003070683541147555, 0.008122796700066756, -0.0003627693785477309, 0.006626775886497746, -0.003637959428330676, -0.004317700956812209, 0.0017917841011632122, -0.009872701256693414, 0.0034875587238231754, \
        -0.006214057833577861, -0.013616401573194577, -0.0047525425525088765, -0.0019547988456830472, -0.000861098620763391, -0.0017324641984695812, 0.009316307361745064, -0.0004748283235934945, -0.003430995879251626, \
        0.0009751332145461938, 0.002596792357855632, 0.008144551540278962, -0.006448853380854076, 0.0005082359779451575, -0.005305324050500932]
        ta_x, ta_y = saved_ta_x[sim_num], saved_ta_y[sim_num]

    elif ta_error=='random':
        # Simulate the TA error from a 5mas normal distribution
        ta_x, ta_y = get_ta_error(error='default')
    elif isinstance(ta_error, (int, float)):
        # Simulate the TA error from an X normal distribution (should provide in arcsec)
        ta_x, ta_y = get_ta_error(error=ta_error)
    elif ta_error=='none':
        ta_x, ta_y = 0., 0.
        fsm_error = 'none'
    else:
        raise ValueError('Target Acquisition (ta_error) string not recognised, options are "random", "saved", or "none", or user specified values.')
    
    sgds = []
    for i, (sx, sy) in enumerate(pointings):
        if i > 0:
            errx, erry = get_fsm_error(error=fsm_error)
            offset_x = sx + errx + ta_x
            offset_y = sy + erry + ta_y
        else:
            offset_x = sx + ta_x
            offset_y = sy + ta_y
        sgds.append([offset_x, offset_y])
    return sgds

def get_ta_error(error='default'):
    ''' 5mas 1-sigma/axis error (~7mas radial)
    '''
    if error == 'default': 
        error = 5e-3
    return np.random.normal(loc=0.,scale=error,size=2)

def get_fsm_error(error='default'):
    '''2mas 1/sigma/axis error from the fine steering mirror
    '''
    if error == 'default':
        error = 2e-3
    elif error == 'none':
        error = 0.

    return np.random.normal(loc=0.,scale=error,size=2)

