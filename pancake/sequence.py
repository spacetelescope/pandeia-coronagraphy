from __future__ import absolute_import, print_function

import os
import json
import numpy as np
from pandeia.engine.calc_utils import build_default_calc
from copy import deepcopy
from collections import OrderedDict

from .utilities import optimise_readout, compute_magnitude, equatorial_to_ecliptic
from .engine import calculate_target, options
from .scene import Scene, create_SGD
from .opds import OPDFile_to_HDUList, OTE_WFE_Drift_Model
from .io import determine_instrument, determine_aperture, determine_subarray, determine_aperture, determine_pixel_scale, sequence_input_checks, determine_exposure_time

from astropy.io import fits
import astropy.units as u
from scipy.interpolate import interp1d

import webbpsf

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
'''
Sequence class used to construct an observational sequence from invidual specified
observations of user defined scenes. 
'''
class Sequence():
    def __init__(self, **kwargs):
        self.observation_sequence = []

    #Add observation does the heavy lifting, with a wide variety of input options. 
    def add_observation(self, scene, exposures, mode='coronagraphy', nircam_mask='default', nircam_subarray='default', miri_subarray='default', telescope='jwst', optimise_margin=0.05, optimize_margin=None, max_sat=0.95, rolls=None, nircam_sgd=None, miri_sgd=None, scale_exposures=None, verbose=True):
        #First copy scene so that the user provided scene isn't modified
        scene = deepcopy(scene)

        #If there is only one exposure, convert it to a list.
        if isinstance(exposures, tuple):
            exposures = [exposures]
        #Check if max roll angle requested. 
        if rolls == 'max':
            # Use an initial roll of 0 (i.e. what the input as the planet PA) and then a second roll of 14 degrees
            rolls = [0, 14]
        #Perform remaining checks on input parameters
        sequence_input_checks(exposures, mode, nircam_mask, nircam_subarray, miri_subarray, telescope, rolls, nircam_sgd, miri_sgd)

        #Loop over each exposure to build independent observation dictionaries. 
        temp_obs_dict_array = []
        for exposure in exposures:
            #Extract filter from exposure
            filt = exposure[0].lower()

            #Identify the instrument, aperture, and subarray from the filter and mode
            instrument = determine_instrument(filt)
            aperture = determine_aperture(filt, nircam_mask, mode)
            subarray = determine_subarray(filt, mode, nircam_subarray, miri_subarray)

            # Construct observation configuration dictionary for Pandeia input
            obs_dict = build_default_calc(telescope, instrument, mode)

            obs_dict['scene'] = scene.pandeia_scene
            obs_dict['scene_name'] = scene.scene_name
            obs_dict['configuration']['detector']['subarray'] = subarray.lower()
            obs_dict['configuration']['instrument']['aperture'] = aperture.lower()
            obs_dict['configuration']['instrument']['filter'] = filt.lower()

            #Extract readout information from exposure settings
            if optimize_margin != None: optimise_margin = optimize_margin #Check for US spelling
            pattern, groups, integrations = self._extract_readout(scene, exposure, subarray, obs_dict, optimise_margin, scale_exposures, max_sat, verbose=verbose)

            obs_dict['configuration']['detector']['readout_pattern'] = pattern.lower()
            obs_dict['configuration']['detector']['ngroup'] = groups
            obs_dict['configuration']['detector']['nint'] = integrations

            #Check if dithers are meant to be performed.
            if instrument == 'nircam' and nircam_sgd != None:
                obs_dict['dither_strategy'] = nircam_sgd
            elif instrument == 'miri' and miri_sgd != None:
                obs_dict['dither_strategy'] = miri_sgd
            else:
                obs_dict['dither_strategy'] = 'SINGLE-POINT'

            temp_obs_dict_array.append(obs_dict)

        #Check if any rolls were requested
        if rolls == None:
            #There are no rolls, just add the obs_dicts to the sequence as they are. 
            for obs_dict in temp_obs_dict_array:
                obs_dict['scene_rollang'] = 0
                self.observation_sequence.append(obs_dict)
        else: 
            #We need to add each of the roll exposures, making sure that things are grouped by the coronagraph used. 
            prev_inst_index = 0
            for i, obs_dict in enumerate(temp_obs_dict_array):
                if i == 0:
                    if len(temp_obs_dict_array) > 1:
                        #First observation, just append with first roll
                        temp_scene = deepcopy(scene)
                        temp_scene.rotate_scene(rolls[0])
                        obs_dict['scene'] = temp_scene.pandeia_scene
                        obs_dict['scene_rollang'] = rolls[0]
                        self.observation_sequence.append(deepcopy(obs_dict)) #Deepcopy otherwise roll will adjust all obs_dicts
                    else:
                        #There is only a single filter in the obs dict array, no coronagraph changes.
                        for roll in rolls:
                            temp_scene = deepcopy(scene)
                            temp_scene.rotate_scene(roll)
                            obs_dict['scene'] = temp_scene.pandeia_scene
                            obs_dict['scene_rollang'] = roll
                            self.observation_sequence.append(deepcopy(obs_dict))
                else:
                    #Up to but excluding the last observation. 
                    prev_aperture = temp_obs_dict_array[i-1]['configuration']['instrument']['aperture']
                    curr_aperture = obs_dict['configuration']['instrument']['aperture']

                    if prev_aperture == curr_aperture:
                        #Coronagraphs are the same, don't want to roll yet. 
                        temp_scene = deepcopy(scene)
                        temp_scene.rotate_scene(roll)
                        obs_dict['scene'] = temp_scene.pandeia_scene
                        obs_dict['scene_rollang'] = rolls[0]
                        self.observation_sequence.append(deepcopy(obs_dict))
                    else:
                        #Coronagraphs must be different, need to append the other rolls now before the switch
                        for roll in rolls[1:]:
                            for temp_obs_dict in temp_obs_dict_array[prev_inst_index:i]:
                                temp_scene = deepcopy(scene)
                                temp_scene.rotate_scene(roll)
                                temp_obs_dict['scene'] = temp_scene.pandeia_scene
                                temp_obs_dict['scene_rollang'] = roll
                                self.observation_sequence.append(deepcopy(temp_obs_dict))
                        #Set an index for the first observation with the next instrument. 
                        prev_inst_index = i

                        #Now append the new coronagraphs first roll
                        temp_scene = deepcopy(scene)
                        temp_scene.rotate_scene(roll)
                        obs_dict['scene'] = temp_scene.pandeia_scene
                        obs_dict['scene_rollang'] = rolls[0]
                        self.observation_sequence.append(deepcopy(obs_dict))

                    #Check if this is the last observation
                    if i == (len(temp_obs_dict_array)-1):
                        #Append all the rolls for the current working coronagraph
                        for roll in rolls[1:]:
                            for temp_obs_dict in temp_obs_dict_array[prev_inst_index:]:
                                temp_scene = deepcopy(scene)
                                temp_scene.rotate_scene(roll)
                                temp_obs_dict['scene'] = temp_scene.pandeia_scene
                                temp_obs_dict['scene_rollang'] = roll
                                self.observation_sequence.append(deepcopy(temp_obs_dict))

            # #We need to add each of the roll exposures, making sure that things are grouped by the coronagraph used. 
            # prev_inst_index = 0
            # for i, obs_dict in enumerate(temp_obs_dict_array):
            #     if i == 0:
            #         if len(temp_obs_dict_array) > 1:
            #             #First observation, just append with first roll
            #             obs_dict['strategy']['scene_rotation'] = rolls[0]
            #             self.observation_sequence.append(deepcopy(obs_dict)) #Deepcopy otherwise roll will adjust all obs_dicts
            #         else:
            #             #There is only a single filter in the obs dict array, no coronagraph changes.
            #             for roll in rolls:
            #                 obs_dict['strategy']['scene_rotation'] = roll
            #                 self.observation_sequence.append(deepcopy(obs_dict))
            #     else:
            #         #Up to but excluding the last observation. 
            #         prev_aperture = temp_obs_dict_array[i-1]['configuration']['instrument']['aperture']
            #         curr_aperture = obs_dict['configuration']['instrument']['aperture']

            #         if prev_aperture == curr_aperture:
            #             #Coronagraphs are the same, don't want to roll yet. 
            #             obs_dict['strategy']['scene_rotation'] = rolls[0]
            #             self.observation_sequence.append(deepcopy(obs_dict))
            #         else:
            #             #Coronagraphs must be different, need to append the other rolls now before the switch
            #             for roll in rolls[1:]:
            #                 for temp_obs_dict in temp_obs_dict_array[prev_inst_index:i]:
            #                     temp_obs_dict['strategy']['scene_rotation'] = roll
            #                     self.observation_sequence.append(deepcopy(temp_obs_dict))
            #             #Set an index for the first observation with the next instrument. 
            #             prev_inst_index = i

            #             #Now append the new coronagraphs first roll
            #             obs_dict['strategy']['scene_rotation'] = rolls[0]
            #             self.observation_sequence.append(deepcopy(obs_dict))

            #         #Check if this is the last observation
            #         if i == (len(temp_obs_dict_array)-1):
            #             #Append all the rolls for the current working coronagraph
            #             for roll in rolls[1:]:
            #                 for temp_obs_dict in temp_obs_dict_array[prev_inst_index:]:
            #                     temp_obs_dict['strategy']['scene_rotation'] = roll
            #                     self.observation_sequence.append(deepcopy(temp_obs_dict))

    def run(self, ta_error='saved', wavefront_evolution=True, on_the_fly_PSFs=False, wave_sampling=3, save_file=False, resume=False, verbose=True, cache='none', cache_path='default' ,offaxis_nircam=[1,1], offaxis_miri=[1,1], debug_verbose=False, initial_wavefront_realisation=4):
        #PanCAKE adjustable options
        pancake_options = options
        pancake_options.verbose = debug_verbose
        pancake_options.wave_sampling = wave_sampling
        pancake_options.on_the_fly_PSFs = on_the_fly_PSFs
        pancake_options.cache = cache 

        if cache != 'none':
            if cache_path == 'default':
                cache_path = os.getcwd() + '/PSF_CACHE/'
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
        else:
            cache_path = None
        pancake_options.cache_path = cache_path


        #Create the HDUList for saving our results to. 
        primary_header = fits.Header()
        hdu_array = [fits.PrimaryHDU(header=primary_header)]
        hdulist = fits.HDUList(hdu_array)

        #Set the index of the first observation to simulate. 
        start_index = 0 

        # If we are saving the results to a file, set this up.
        if save_file != False:
            if resume != True:
                #If we pass the checks, overwrite any existing file with the same name. 
                #Check provided directory exists and is the right format before running all the simulations
                save_dir = '/'.join(save_file.split('/')[0:-1])
                if not os.path.exists(save_dir):
                    raise IOError('Provided save directory "{}" not found.'.format(save_dir))
                elif save_file.split('.')[-1] != 'fits':
                    raise IOError('Provided save file "{}" not of "*.fits" format.'.format(save_file))

                hdulist.writeto(save_file, overwrite=True) 
            else:
                #Need to carry on from where a saved simulation last ended. Figure out where that is...
                with fits.open(save_file) as save_hdul:
                    if len(save_hdul)-1 == len(self.observation_sequence):
                        #All simulations already completed.
                        print('WARNING: All observation simulations have been completed, nothing to resume.')
                        return
                    else:
                        print('Resuming previous observation sequence...')
                        #Change the start index to the correct value. 
                        start_index = len(save_hdul)-1

        #Calculate OPD's throughout the observation if requested and on_the_fly_PSFs are being used. 
        if wavefront_evolution == True and on_the_fly_PSFs == True and len(self.observation_sequence) > 1:
            if verbose: print('Computing OPD Maps...')
            opd_res = self._calculate_opds(opd_realisation=initial_wavefront_realisation)
            if opd_res == None: 
                wavefront_evolution = False
            else:
                nircam_opds, miri_opds = opd_res
        else:
            #Don't use an OPD drift if we aren't using on_the_fly_PSFs or only a single observation. 
            wavefront_evolution = False

        #Next, we loop over all the observations within the sequence and simulate each one individually. 
        if verbose: print('Running Simulations...')
        observation_counter = 0
        unocculted_images = []
        for i_raw, base_obs_dict in enumerate(self.observation_sequence[start_index:]):
            i = i_raw + start_index
            scene_name = base_obs_dict['scene_name']
            filt = base_obs_dict['configuration']['instrument']['filter']
            instrument = base_obs_dict['configuration']['instrument']['instrument']

            if instrument == 'miri':
                offaxis_x, offaxis_y = offaxis_miri
            elif instrument == 'nircam':
                offaxis_x, offaxis_y = offaxis_nircam

            if verbose: print('--> Observation {}/{} // {}_{}'.format(i+1, len(self.observation_sequence), scene_name, filt.upper()))

            ##### First let's calculate an unocculted image to compute the contrast curve with
            offaxis_dict = deepcopy(base_obs_dict)
            for j in range(len(offaxis_dict['scene'])):
                #Offset all the sources 
                offaxis_dict['scene'][j]['position']['x_offset'] += offaxis_x
                offaxis_dict['scene'][j]['position']['y_offset'] += offaxis_y

            # Ignore any saturation, allowing counts above full well, and turn off on_the_fly_PSFs for speed. 
            pancake_options.set_saturation(False)
            pancake_options.on_the_fly_PSFs = False

            # Ensure the scene isn't rotated to make extracting the PSF center easier
            offaxis_dict['strategy']['scene_rotation'] = 0 
            
            # Calculate off-axis image. 
            offaxis_result = calculate_target(offaxis_dict)
            
            # Return settings to original values
            pancake_options.set_saturation(True)
            pancake_options.on_the_fly_PSFs = on_the_fly_PSFs

            ##### Now we will compute the actual observations
            # Assemble small grid dither array. Even without any SGD this will happen, it will just use a single dither point. 
            # This is also where the random target acquisition error is added.
            sgds = create_SGD(ta_error=ta_error, pattern_name=base_obs_dict['dither_strategy'], sim_num=i)

            for j, sgd in enumerate(sgds):
                if len(sgds) > 1:
                    #Print the small grid dither step we are on
                    if verbose: print('----> Small Grid Dither {}/{}'.format(j+1,len(sgds)))

                #Begin by making a new copy of the obs_dict so that shifts in source offsets only happen once. 
                obs_dict = deepcopy(base_obs_dict)

                #Offset all the sources in the scene by the necessary dither + target acquisition amounts. 
                xoff, yoff = sgd
                for k in range(len(obs_dict['scene'])):
                    obs_dict['scene'][k]['position']['x_offset'] += xoff
                    obs_dict['scene'][k]['position']['y_offset'] += yoff

                if wavefront_evolution == True:
                    #Use the calculated OPD for this specific observation. 
                    if instrument == 'nircam':
                        options.on_the_fly_webbpsf_opd = nircam_opds[observation_counter] 
                    elif instrument == 'miri':
                        options.on_the_fly_webbpsf_opd = miri_opds[observation_counter]

                ######## RUN OBSERVATION ########
                data = calculate_target(obs_dict) 
                #################################

                if j == 0:
                    #We are on the first simulation, so lets make the right size array now.
                    first_detector_image = data['2d']['detector']
                    # Shape is SGDs + 1 as we also need to include the unocculted image from earlier. 
                    detector_images = np.empty([len(sgds)+1, first_detector_image.shape[0],  first_detector_image.shape[1]])
                    detector_images[0] = first_detector_image
                else:
                    #We are on a simulation due to dithering, add it to the already defined array
                    detector_images[j] = data['2d']['detector']

                observation_counter += 1

            ##### Append the unocculted image to the array
            detector_images[-1] = offaxis_result['2d']['detector']

            #We need to create a HDU for this observation, first make the header
            image_header = fits.Header()
            image_header['EXTNAME'] = str(i+1)+ ':' + base_obs_dict['scene_name'] + '_' + base_obs_dict['configuration']['instrument']['aperture'].upper() + '_' + base_obs_dict['configuration']['instrument']['filter'].upper()
            image_header['TARGET'] = base_obs_dict['scene_name']
            image_header['INSTRMNT'] = base_obs_dict['configuration']['instrument']['instrument']
            image_header['FILTER'] = base_obs_dict['configuration']['instrument']['filter']
            image_header['APERTURE'] = base_obs_dict['configuration']['instrument']['aperture']
            image_header['SUBARRAY'] = base_obs_dict['configuration']['detector']['subarray']
            image_header['PATTERN'] = base_obs_dict['configuration']['detector']['readout_pattern']
            image_header['NGROUP'] =  base_obs_dict['configuration']['detector']['ngroup'] 
            image_header['NINT'] = base_obs_dict['configuration']['detector']['nint']
            image_header['TEXP'] = (determine_exposure_time(image_header['SUBARRAY'], image_header['PATTERN'], image_header['NGROUP'], image_header['NINT']), 'seconds')
            image_header['PIXSCALE'] = (determine_pixel_scale(image_header['INSTRMNT'], image_header['FILTER']), 'arcsec/pixel')
            #image_header['ROLLANG'] = (base_obs_dict['strategy']['scene_rotation'], 'degrees')
            image_header['ROLLANG'] = (base_obs_dict['scene_rollang'], 'degrees')
            image_header['DITHER'] = base_obs_dict['dither_strategy']
            
            image_header['NSOURCES'] = len(base_obs_dict['scene'])
            for source in base_obs_dict['scene']:
                for j, sgd in enumerate(sgds):
                    image_header['SOURCE{}'.format(source['id'])] = source['pancake_parameters']['name']
                    image_header['S{}XOFF{}'.format(source['id'],j+1)] = source['position']['x_offset'] + sgd[0]
                    image_header['S{}YOFF{}'.format(source['id'],j+1)] = source['position']['y_offset'] + sgd[1]
                image_header['S{}OFFAX'.format(source['id'])] = source['position']['x_offset'] + offaxis_x
                image_header['S{}OFFAY'.format(source['id'])] = source['position']['y_offset'] + offaxis_y
                image_header['S{}VGAMG'.format(source['id'])]  = compute_magnitude(source['spectrum']['sed']['spectrum'][0], source['spectrum']['sed']['spectrum'][1], base_obs_dict['configuration']['instrument']['filter'])
                image_header.add_blank('', before='SOURCE{}'.format(source['id']))

            image_header['PCOTFPSF'] = on_the_fly_PSFs
            image_header['PCWAVSAM'] = (wave_sampling, 'Only valid if PCOTFPSF==True')
            image_header['PCOPDDRF'] = wavefront_evolution
            image_header.add_blank('', before='PCOTFPSF')

            # Append the images and header information to the initial HDUList. 
            image_data = detector_images
            hdulist.append(fits.ImageHDU(image_data, header=image_header))

            if save_file != False:
                # Then we also want to append the data to our saved file, doing it this way means if there is a crash
                # halfway through then things are still saved.  
                fits.append(save_file, image_data, image_header)

        return hdulist

    def _relative_exposure_time(self, scene, filt, master_scene, master_exposure_time=3600):
        #Figure out how much longer (or shorter) an observation should be with reference to the relative flux of the
        #brightest sources between two scenes. Inaccurate if the flux of the brightest source isn't >> than other sources. 
        magnitude=50
        for i in range(len(scene.source_list)):
            source = scene.pandeia_scene[i]
            spec_wave, spec_flux = source['spectrum']['sed']['spectrum']
            temp_magnitude = compute_magnitude(spec_wave, spec_flux, filt)
            if temp_magnitude < magnitude:
                magnitude = temp_magnitude

        master_magnitude=50
        for i in range(len(master_scene.source_list)):
            source = master_scene.pandeia_scene[i]
            spec_wave, spec_flux = source['spectrum']['sed']['spectrum']
            temp_magnitude = compute_magnitude(spec_wave, spec_flux, filt)
            if temp_magnitude < master_magnitude:
                master_magnitude = temp_magnitude

        flux_ratio = 100**((magnitude-master_magnitude)/5)
        exposure_time = flux_ratio * master_exposure_time

        return exposure_time

    def _extract_readout(self, scene, exposure, subarray, obs_dict, optimise_margin, scale_exposures, max_sat, verbose=True):
        filt = exposure[0].lower()
        if exposure[1] in ['optimise', 'optimize']:
            #Need to optimise the exposure readouts
            if verbose: print('Optimising Readout // {} // Exposure: {}, {} seconds'.format(obs_dict['scene_name'], exposure[0].upper(), str(exposure[2])))
            exposure_time = exposure[2]
            if scale_exposures != None:
                if isinstance(scale_exposures, (int, float)):
                    #Scale exposure time by a numeric value. 
                    if verbose: print('--> Scaling provided exposure times by {}'.format(scale_exposures))
                    exposure_time *= scale_exposures
                elif isinstance(scale_exposures, Scene):
                    #Scale exposure time to match flux of another scene. 
                    if verbose: print('--> Scaling provided exposure times by relative flux of: "{}"'.format(scale_exposures.scene_name))
                    master_scene = scale_exposures
                    exposure_time = self._relative_exposure_time(scene, filt, master_scene, master_exposure_time=exposure_time)
                else:
                    raise ValueError('Chosen "scale_exposures" setting not recognised. Select int/float scaling factor or defined Scene.')
            
            pattern, groups, integrations = optimise_readout(obs_dict, exposure_time, optimise_margin, max_sat=max_sat)
            exposure_time = determine_exposure_time(subarray, pattern, groups, integrations)
            #Notify user of the optimised readout parameters
            if verbose: print('--> Pattern: {}, Number of Groups: {}, Number of Integrations: {} = {}s'.format(pattern.upper(), groups, integrations, int(exposure_time+0.5)))
        else:
            #Parameters have been specified explicitly
            if scale_exposures != None:
                exposure_time = determine_exposure_time(subarray, exposure[1], exposure[2], exposure[3])
                if isinstance(scale_exposures, (int, float)):
                    #Scale readout parameters by a numeric value
                    if verbose: print('--> Scaling provided exposure times by {}'.format(scale_exposures))
                    exposure_time *= scale_exposures
                elif isinstance(scale_exposures, Scene):
                    #Scale readout to match the flux of another scene. 
                    if verbose: print('--> Scaling provided exposure times by relative flux of: "{}"'.format(scale_exposures.scene_name))
                    master_scene = scale_exposures
                    exposure_time = self._relative_exposure_time(scene, filt, master_scene, master_exposure_time=exposure_time)
                else:
                    raise ValueError('Chosen "scale_exposures" setting not recognised. Select int/float scaling factor or defined Scene.')

                #"Re"-optimise readout patterns using the new exposure time 
                pattern, groups, integrations = optimise_readout(obs_dict, exposure_time, optimise_margin, max_sat=max_sat)
                exposure_time = determine_exposure_time(subarray, pattern, groups, integrations)
                #Notify user of the optimised readout parameters
                if verbose: print('---> Pattern: {}, Number of Groups: {}, Number of Integrations: {} = {}s'.format(pattern.upper(), groups, integrations, int(exposure_time+0.5)))
            else:
                pattern, groups, integrations = exposure[1:]
        return pattern, groups, integrations

    def _calculate_opds(self, opd_estimate='requirements', opd_realisation=1, pa_range='median'):
        '''
        We want to calculate a range of optical path difference (OPD) maps dependent on how the observatory
        moves throughout the sequenced observations. 
        '''

        # First thing we need to do is gather the geometric properties of the scenes in the sequence so that we 
        # know what the ra, dec, lambda, beta, and pitch angle should be for each observation. 
        geom_props = self._get_geom_props(pa_range=pa_range)
        # Check that the properties were received correctly, if not we can't perform the OPD drift 
        if geom_props == None: 
            return None

        obs_times = [0]
        pitch_angles = [geom_props[self.observation_sequence[0]['scene_name']]['pitch_angle']]
        skip_indices = [0] #OPDs that we don't need for the sequence, but are needed to more accurately estimate the OPD changes overall
        #Loops over observations to identify changes in observation time and slews
        for i, obs_dict in enumerate(self.observation_sequence):
            if i != 0:
                ###########This is the second observation onwards, potential for a slew 
                #####First, find out if we have slewed to a different scene. 
                previous_scene = self.observation_sequence[i-1]['scene_name']
                current_scene = self.observation_sequence[i]['scene_name']
                if previous_scene != current_scene:
                    #Need to determine how large a slew

                    #Calculate the angular offset between the two scenes
                    #Let's work in an ecliptic coordinate system rather than equatorial, where lamb and beta are the
                    #ecliptic longitudes and latitudes respectively. 
                    previous_lamb, previous_beta = geom_props[previous_scene]['lamb'], geom_props[previous_scene]['beta']
                    current_lamb, current_beta = geom_props[current_scene]['lamb'], geom_props[current_scene]['beta']

                    previous_lamb_rad, previous_beta_rad = np.deg2rad(previous_lamb), np.deg2rad(previous_beta) 
                    current_lamb_rad, current_beta_rad = np.deg2rad(current_lamb), np.deg2rad(current_beta)

                    #Absolute slew is just the distance between the two points
                    scene_slew_angdist_rad = np.arccos(np.sin(previous_beta_rad)*np.sin(current_beta_rad) + \
                        np.cos(previous_beta_rad)*np.cos(current_beta_rad)*np.cos(previous_lamb_rad-current_lamb_rad))
                    scene_slew_angdist = scene_slew_angdist_rad * (180/np.pi)

                    scene_pitch_angle = geom_props[current_scene]['pitch_angle']
                else:
                    #There is no slew between scenes
                    scene_slew_angdist = 0
                    scene_pitch_angle = geom_props[previous_scene]['pitch_angle']
                
                ##### Now find out if the instrument has changed. 
                previous_inst = self.observation_sequence[i-1]['configuration']['instrument']['instrument']
                current_inst = self.observation_sequence[i]['configuration']['instrument']['instrument']
                if previous_inst != current_inst:

                    '''
                    The instrument has changed, so some time will have passed to perform the slew. In reality
                    this depends on the exact subarray / aperture being used. Here we approximate to 500" 
                    based on the overall offset between NIRCam and MIRI from:

                    https://jwst-docs.stsci.edu/jwst-observatory-hardware/jwst-field-of-view

                    This is roughly equivalent to 10 minutes of slew time. 
                    '''
                    if sorted([previous_inst, current_inst]) == ['miri', 'nircam']:
                        inst_slew_angdist = 500 / 3600 #Potential for this to be negative, but assume positive for now. 
                    else:
                        print('WARNING: Unable to process angular distance change between instruments.')

                    #From the same link above, change in pitch angle (V3) is ~100", also could be negative.
                    if current_inst == 'miri': 
                        inst_delta_pitch_angle = -100 / 3600
                    elif current_inst == 'nircam':
                        inst_delta_pitch_angle = 100 / 3600
                else:
                    inst_slew_angdist = 0
                    inst_delta_pitch_angle = 0
                
                if inst_slew_angdist != 0 or scene_slew_angdist !=0 : 
                    #There has been a slew
                    #Now turn a slew distance into a slew time. 
                    slew_angdist = scene_slew_angdist + inst_slew_angdist
                    pitch_angle = scene_pitch_angle + inst_delta_pitch_angle

                    slew_time = self._slew_angdist_to_time(slew_angdist)

                    obs_times.append(obs_times[-1] + slew_time)
                    pitch_angles.append(pitch_angle)
                    #Log the index so that we don't use its OPD for later calculations
                    skip_indices.append(len(pitch_angles)-1)

            #Now that slews are accounted for, append observations as normal
            if obs_dict['dither_strategy'] == 'SINGLE-POINT':
                dither_points = 1
            else:
                dither_points = int(obs_dict['dither_strategy'][0])

            for j in range(dither_points):
                subarray = obs_dict['configuration']['detector']['subarray']
                pattern = obs_dict['configuration']['detector']['readout_pattern']
                groups =  obs_dict['configuration']['detector']['ngroup'] 
                integrations = obs_dict['configuration']['detector']['nint']
                
                exposure_time = determine_exposure_time(subarray, pattern, groups, integrations) 

                #Append the centre time of the observation, these are the OPD's we actually want.
                pitch_angles.append(pitch_angles[-1])
                obs_times.append(obs_times[-1] + (exposure_time / 2))

                #Also add the end observation time, but make sure it is skipped.
                pitch_angles.append(pitch_angles[-1])
                obs_times.append(obs_times[-1] + (exposure_time / 2))
                skip_indices.append(len(pitch_angles)-1) 

        # Now that we know the pitch angle at specific times throughout our observation, we can load in the base OPD
        # files from WebbPSF, evolve them, delete the ones we don't need, and return the specific OPD for each observation.
        all_opd_estimates = ['predicted', 'requirements']
        if opd_estimate not in all_opd_estimates:
            raise ValueError('Chosen OPD estimate "{}" not recognised. Compatible options are : {}'.format(opd_estimate, ', '.join(all_opd_estimates)))

        base_opd =  'OPD_RevW_ote_for_{}_'+opd_estimate+'.fits.gz'

        #Get Base OPD for NIRCam and MIRI (or just one if only one in sequence).
        nircam_opd_file = os.path.join(webbpsf.utils.get_webbpsf_data_path(), 'NIRCam', 'OPD', base_opd.format('NIRCam'))
        miri_opd_file = os.path.join(webbpsf.utils.get_webbpsf_data_path(), 'MIRI', 'OPD', base_opd.format('MIRI'))

        nircam_opd_hdu = OPDFile_to_HDUList(nircam_opd_file, slice_to_use=opd_realisation)
        miri_opd_hdu = OPDFile_to_HDUList(miri_opd_file, slice_to_use=opd_realisation)

        BaseNIRCamOPD = OTE_WFE_Drift_Model(opd=nircam_opd_hdu)
        BaseMIRIOPD =  OTE_WFE_Drift_Model(opd=miri_opd_hdu)

        nircam_opd_hdul = BaseNIRCamOPD.opds_as_hdul(obs_times*u.second, np.array(pitch_angles), slew_averages=False)
        miri_opd_hdul = BaseMIRIOPD.opds_as_hdul(obs_times*u.second, np.array(pitch_angles), slew_averages=False)

        #Remove OPD's that we no longer need, leaving just those at the centre of each of our observations. 
        for index in sorted(skip_indices, reverse=True):
            del nircam_opd_hdul[index]
            del miri_opd_hdul[index]

        #Now need to restructure as a list of HDULists for integration with WebbPSF
        nircam_opd_lhdul = [fits.HDUList([hdu]) for hdu in nircam_opd_hdul]
        miri_opd_lhdul = [fits.HDUList([hdu]) for hdu in miri_opd_hdul]

        return nircam_opd_lhdul, miri_opd_lhdul

    def _get_geom_props(self, pa_range='median'):
        ##### First get the RA, Dec
        geom_props = {}
        for obs_dict in self.observation_sequence[:]:
            scene_name = obs_dict['scene_name'] 
            if scene_name not in geom_props.keys():
                geom_props[scene_name] = {}

                ra, dec = self._find_ra_dec(obs_dict)

                if None in [ra, dec]:
                    geom_props[scene_name]['failed_flag'] = True
                else:
                    geom_props[scene_name]['failed_flag'] = False
                    geom_props[scene_name]['ra'] = ra
                    geom_props[scene_name]['dec'] = dec 
                
        ##### Now need to check whether all RA's or Dec's have values
        scene_names = list(geom_props.keys())
        flags = [geom_props[scene_name]['failed_flag'] for scene_name in scene_names]

        if True in flags:
            #Something failed
            if all(x == True for x in flags):
                #They all failed
                base_ra = 0
                base_dec = 45
            else:
                #At least one didn't fail 
                base_ra = geom_props[scene_names[flags.index(False)]]['ra']
                base_dec = geom_props[scene_names[flags.index(False)]]['dec']

            for i, scene_name in enumerate(scene_names):
                if flags[i] == True:
                    ra = base_ra + 5*i  
                    dec =  base_ra + 5*i
                    geom_props[scene_name]['ra'] = ra
                    geom_props[scene_name]['dec'] = dec 
                    print('WARNING: RA and/or Dec value not present for Scene "{}", assuming values of RA={} and Dec={}'.format(scene_name, ra, dec))
        #Calculate ecliptic longitude (lamb) and latitude (beta)
        for i, scene_name in enumerate(scene_names):
            lamb, beta = equatorial_to_ecliptic(geom_props[scene_name]['ra'], geom_props[scene_name]['dec'], form='degrees')
            geom_props[scene_name]['lamb'] = lamb
            geom_props[scene_name]['beta'] = beta 

        ##### Get the pitch angles for all of the scenes. 
        #Look at every orientation throughout the year
        orientations = np.arange(0, 360, 0.1)
        all_pitch_angles = np.empty((len(orientations), len(geom_props.keys())))
        all_pitch_angle_ranges = np.empty(orientations.shape)
        for i, ostep in enumerate(orientations):
            temp_pitch_angles = []
            #Loop over the scene names to preserve the order for later
            for scene_name in scene_names:

                #Get the ecliptic latitude and longitude in radians
                beta_rad = geom_props[scene_name]['beta'] * (np.pi / 180)
                lamb_rad = ((geom_props[scene_name]['lamb'] + ostep) % 360 ) * (np.pi / 180)
    
                #Identify the latitude after a rotation of the coordinate system by 90 degrees
                #--> This is equivalent to the pitch angle: AC has saved the derivation to DayOne on Feb 16 2021. 
                pitch_angle = np.arcsin(np.cos(beta_rad)*np.sin(lamb_rad)) * (180/np.pi)
                temp_pitch_angles.append(pitch_angle)
            
            #Save the pitch angles
            all_pitch_angles[i] = temp_pitch_angles

            if all([-5 < pa < 45 for pa in temp_pitch_angles]):
                #Then at this orientation all scenes can be slewed to
                pitch_angle_range = max(temp_pitch_angles) - min(temp_pitch_angles)
                all_pitch_angle_ranges[i] = pitch_angle_range
            else:
                #It is not possible to observe everything at this slew
                all_pitch_angle_ranges[i] = np.nan

        #Need to check if any of the orientations were viable. 
        if np.isnan(all_pitch_angle_ranges).all():
            #There is no orientation where all scenes lie within the FOV.
            print('WARNING: Specified scenes always span a pitch angle range of greater than 50 degrees and cannot be scheduled in sequence. No OPD drift can be calculated for this simulation.')
            return None

        # Now we can use the pitch angle range to specify what the PA location of each Scene is for this sequence. 
        if pa_range == 'minimum':
            pa_index = np.nanargmin(all_pitch_angle_ranges)
        elif pa_range == 'maximum':
            pa_index = np.nanargmax(all_pitch_angle_ranges)
        elif pa_range == 'median':
            valid_pa_ranges = all_pitch_angle_ranges[np.logical_not(np.isnan(all_pitch_angle_ranges))]
            median_pa_range = np.sort(valid_pa_ranges)[len(valid_pa_ranges)//2]
            pa_index = np.where(all_pitch_angle_ranges == median_pa_range)[0][0]

        #Set the pitch angles for this Sequence
        pitch_angles = all_pitch_angles[pa_index]
        for i, scene_name in enumerate(scene_names):
            geom_props[scene_name]['pitch_angle'] = pitch_angles[i] 

        return geom_props

    def _find_ra_dec(self, obs_dict):
        #Search for RA and DEC values
        ra, dec = None, None
        for source in obs_dict['scene']:
            try: 
                ra = source['pancake_parameters']['ra']
                dec = source['pancake_parameters']['dec']
            except:
                #These values haven't been assigned. 
                continue
        return ra, dec

    def _slew_angdist_to_time(self, slew_angdist):
        #Slew distance must be in degrees. 
        
        #Slew distance and time costs taken from https://jwst-docs.stsci.edu/jppom/visit-overheads-timing-model/slew-times
        slew_angdists = np.array([0.0000000, 0.0600000, 0.0600001, 15.0000000, 20.0000000, 20.0000001, 30.0000000, 50.0000000, 100.0000000,
            150.0000000, 300.0000000, 1000.0000000, 3600.0000000, 4000.0000000, 10000.0000000, 10800.0000000, 10800.0000001, 14400.0000000,
            18000.0000000, 21600.0000000, 25200.0000000, 28800.0000000, 32400.0000000, 36000.0000000, 39600.0000000, 43200.0000000,
            46800.0000000, 50400.0000000, 54000.0000000, 57600.0000000, 61200.0000000, 64800.0000000, 68400.0000000, 72000.0000000,
            108000.0000000, 144000.0000000, 180000.0000000, 216000.0000000, 252000.0000000, 288000.0000000, 324000.0000000, 360000.0000000,
            396000.0000000, 432000.0000000, 468000.0000000, 504000.0000000, 540000.0000000, 576000.0000000, 612000.0000000, 648000.0000000])
        slew_times = np.array([0.000, 0.000, 20.480, 20.480, 23.296, 101.632, 116.224, 137.728, 173.568, 198.656, 250.112, 373.504, 572.416,
            592.896, 804.864, 825.600, 521.216, 578.048, 628.608, 674.560, 716.928, 756.608, 793.856, 829.184, 862.848, 894.976, 925.824,
            955.648, 984.320, 1012.224, 1039.104, 1065.344, 1090.816, 1115.648, 1336.448, 1537.408, 1744.000, 1939.328, 2112.192, 2278.272,
            2440.320, 2599.936, 2757.632, 2914.240, 3069.888, 3224.832, 3379.328, 3533.376, 3687.104, 3840.512])

        slew_angdists_deg = slew_angdists /3600 
        time_angdist_interp = interp1d(slew_angdists_deg, slew_times)

        slew_time = time_angdist_interp(slew_angdist)

        return slew_time


    def _get_ordered_scene_names(self, duplicates=True):
        scene_names = []
        for obs_dict in self.observation_sequence:
            scene_names.append(obs_dict['scene_name'])
        if duplicates == False:
            scene_names = list(OrderedDict.fromkeys(scene_names))
        return scene_names

def load_run(save_file):
    #Load in a recently performed and saved run
    with fits.open(save_file) as save_data:
        hdul = deepcopy(save_data)
    return hdul