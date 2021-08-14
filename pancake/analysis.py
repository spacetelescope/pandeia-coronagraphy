import os
import warnings
import re
import numpy as np
import scipy 
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d, interp2d
from scipy.stats import t
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import partial
import json

from .io import read_coronagraph_transmission, determine_bar_offset
import pyklip.klip 
import pyklip.instruments.Instrument as Instrument
import pyklip.parallelized as parallelized
import pyklip.rdi as rdi
import pyklip.fakes as fakes


# Getting erros when trying to use multiprocessing with pyKLIP
# Disable using the below
parallelized.debug = True

##############################
# DISCLAIMER 
# Many functions in this file are heavily inspired by and adapted from the work of Jea Adams on ExoPix.
# https://exopix.readthedocs.io/en/latest/
##############################

warnings.simplefilter('ignore', RuntimeWarning)
def enable_runtime_warnings(state):
	"""
	Function to toggle on/off RuntimeWarning's. Many of these do not impact the functionality of the code
	and therefore can be safely ignored for the vast majority of user cases. As a result, this function is
	immediately called a few lines above. 

	Parameters
		state : bool
			Whether the RuntimeWarning's should be enabled (True), or disabled ( False).
	Returns:
		None
	""" 
	if state==True:
		warnings.simplefilter('always', RuntimeWarning)
	elif state==False:
		warnings.simplefilter('ignore', RuntimeWarning)
	else:
		raise ValueError('Runtime Warnings can only be enabled/disabled with a boolean True/False input.')

#### Impossible to pass the mask argument into pyKLIP, to use with inject_planet() function must use the functools partial function to set the mask variable
def transmission_corrected(input_stamp, input_dx, input_dy, filt, mask, mode='multiply'):
	"""
	Function to apply a 2-dimensional JWST coronagraphic transmission map to an input image.

	Parameters
		input_stamp : 2D ndarray
			Input image, should have dimensions equal to or smaller than the array for the
			coronagraphic transmission map.
		input_dx : 2D ndarray
			Array of X pixel offsets for each element in the array relative to the central pixel of the simulation. 
		input_dy : 2D ndarray
			Array of Y pixel offsets for each element in the array relative to the central pixel of the simulation. 	
		filt : str
			JWST filter string, used to obtain offsets for the NIRCam bar masks. 
		mask : str
			JWST coronagraphic mask string, used to identify which transmission map to apply.
		mode : str
			Whether to 'multiply' or 'divide' the input stamp by the transmission map. 
	Returns
		output_stamp : 2D ndarray
			Equivalent to the input_stamp following the application of the transmission map. 
	"""
	##### Get the x- and y- dimension for the input image
	input_x, input_y = input_stamp.shape 

	##### Read in the transmission array for the mask we are using
	transmission = read_coronagraph_transmission(mask)

	##### If we are using a NIRCam bar mask, the center of the input images will correspond to different 
	##### locations in the transmission map dependent on the filter used.
	#### NOTE: AS OF 14/06/2021, these offsets are not always correct.
	if mask == 'MASKSWB':
		xoff = determine_bar_offset(filt) / 0.0311 #Make sure to convert from arcsec to pixels
		yoff = 0
	elif mask == 'MASKLWB':
		#Important to adjust the sign for the LWB as things are reversed in Pandeia
		xoff = -determine_bar_offset(filt) / 0.063 #Make sure to convert from arcsec to pixels
		yoff = 0
	else:
		xoff, yoff = 0, 0

	##### Now need to get the portion of the transmission map that corresponds to the input stamp image. 
	trans_x, trans_y = transmission.shape
	trans_dx = np.arange(-(trans_x-1)/2, (trans_x)/2) 
	trans_dy = np.arange(-(trans_y-1)/2, (trans_y)/2) 

	#Create interpolation for the tranmission map we are using 
	trans_interp  = interp2d(trans_dx, trans_dy, transmission)

	#Use the interpolation to identify the transmission at each pixel in the input image. 
	transmission_stamp = trans_interp(input_dx[0] + xoff, input_dy.transpose()[0]+ yoff)

	##### Apply the transmission, dependent on which mode has been selected
	if mode == 'multiply':
		output_stamp = input_stamp * transmission_stamp
	elif mode == 'divide':
		output_stamp = input_stamp / transmission_stamp

	return output_stamp

def identify_primary_sources(pancake_results, target, references=None, target_primary_source='default', reference_primary_sources='default'):
	"""
	Function to identify, or assume, the primary sources (i.e. central 'stars') of output PanCAKE simulation results. 

	Parameters
		pancake_results : HDUList
			Simulated results as returned by pancake.sequence.Sequence().run()
		target : str
			The provided string name for the target scene in the observation sequence. 
		references : str / list of strings / NoneType
			The provided string name(s) for the reference scene(s) in the observations sequence, if any.
		target_primary_source : str
			Desired primary source to use for the target scene, or 'default' to assume primary source. 
		reference_primary_sources : str / list of strings
			Desired primary source(s) to use for the reference scene(s), or 'default' to assume primary source(s).
	Returns
		primary_sources : list of strings
			List of the primary source(s), where the source in the '0' index always corresponds to the target scene. 

	"""
	#Get all of the observations names for this simulation
	obs_names = [pancake_results[i].header['EXTNAME'] for i in range(1,len(pancake_results))]
	scene_names = list(dict.fromkeys([i.split('_')[0].split(':')[-1] for i in obs_names]))

	# If references is None, just set references to an empty array
	if references == None:
		references = []

	# Initialise array
	primary_sources = []
	for scene in [target]+references: 
		# Grab only the observations of this scene
		match_obs = [i for i in obs_names if ':'+scene+'_' in i]
		if not match_obs:
			#There were no matches to the input target/reference string
			raise ValueError("Unable to find specified scene '{}' within simulated results. Possible scenes include: {}".format(scene, ', '.join(scene_names)))

		# If error is not raised, we have a match for this scene. 
		# Doesn't really matter which precise observation we use, so just use the first one. 
		example_header = pancake_results[match_obs[0]].header
		num_sources = example_header['NSOURCES']
		all_sources = [example_header['SOURCE{}'.format(i+1)] for i in range(num_sources)]
		if scene == target and target_primary_source != 'default':
			# We need to use the user provided target primary source
			if not target_primary_source in all_sources:
				raise ValueError("Specified source '{}' for target scene '{}' not found. Available sources are: {}".format(target_primary_source, target, ', '.join(all_sources)))
			else:
				primary_source = target_primary_source
		elif scene in references and reference_primary_sources != 'default':
			# We need to use the user provided reference primary source
			ref_source = reference_primary_sources[references.index(scene)]
			if not ref_source in all_sources:
				raise ValueError("Specified source '{}' for reference scene '{}' not found. Available sources are: {}".format(target_primary_source, target, ', '.join(all_sources)))
			else:
				primary_source = ref_source
		else:
			# We choose a default primary source of 'SOURCE1'
			primary_source = example_header['SOURCE1']
			if num_sources != 1:
				# Warn the user which source we picked. 
				print('WARNING: Assuming primary source "{}" for scene "{}"'.format(primary_source, scene))

		#Append primary source to array. Eventual order will be the target primary, then the reference primaries in the order they were provided. 
		primary_sources.append(primary_source)

	return primary_sources

def extract_simulated_images(pancake_results, observations, primary_sources, all_rolls, references=None, extract_offaxis=False, filename_prefix='image'):
	"""
	Function to extract a subset of simulated images from the output of a PanCAKE simulation into a more flexible format. 
	
	Parameters
		pancake_results : HDUList
			Simulated results as returned by pancake.sequence.Sequence().run()
		observations : list of strings
			List of observation strings that correspond to extension names in the pancake_results HDUList
		primary_sources : list of strings
			List of the primary source(s) as obtained by identify_primary_sources
		all_rolls : list of ints / floats
			Which PA roll images to be extracted
		references : str / list of strings / NoneType
			The provided string name(s) for the reference scene(s) in the observations sequence, if any. Can
			not be used in conjunction with retrieving simulated target images. 
		extract_offaxis : bool
			Boolean choice of whether to extract offaxis images or not
		filename_prefix : str
			Simple prefix string used to assign a unique name to each simulated image.
	Returns
		extracted : dict
			Dictionary containing all requested images, their roll angles, center points, assigned filenames, and
			if requested, a 20x20 pixel offaxis PSF stamp image and its peak flux.  
	"""

	##### Create placeholder variables to append to / adjust later. 
	images, pas, centers, filenames = [], [], [], [] 
	offaxis_image = np.array([]) 
	offaxis_psf_stamp, offaxis_peak_flux = None, None 
	
	##### Loop over the requested observations. 
	for i, obs in enumerate(observations): 
		#Extract data and header informatino
		data = pancake_results[obs].data
		head = pancake_results[obs].header
		
		rollang = head['ROLLANG'] # PA roll angle
		pixel_scale = head['PIXSCALE'] # Pixel scale for this observations
		
		filt = obs.split('_')[-1]
		wavelength = float(re.findall('\\d+', filt)[0]) / 1e8
		lambda_d_arcsec = (wavelength / 6.5) * (180 / np.pi) * 3600
		lambda_d_pixel = lambda_d_arcsec / pixel_scale 

		source_keys = ['SOURCE{}'.format(j+1) for j in range(head['NSOURCES'])]
		sources = [head[key] for key in source_keys] # Names of the sources in this observation. 

		if references == None: 
			# This extraction is for target observations.
			primary_source_id = source_keys[sources.index(primary_sources[0])][-1]
			rolls = all_rolls
		else:
			# This extraction is for multiple different reference observations, need to change properties based on specific obs
			ref_scene_index = references.index(obs.split('_')[0].split(':')[-1])
			primary_source_id = source_keys[sources.index(primary_sources[1+ref_scene_index])][-1]
			rolls = all_rolls[ref_scene_index] # Just get the rolls for this particular scene. 

		# Loop over the PA rolls requested. 
		if rollang in rolls: 
			# Loop over each simulated image for this observation. 
			for j, image in enumerate(data):
				if j != len(data)-1:
					#Final index is for offaxis images, all indexes before this are what we're interested in. 

					# Grab X and Y offsets of the primary source.
					# NOTE: Y offset must have its sign reversed due to differences between Pandeia scene construction and analysis of the image arrays
					xoff = head['S{}XOFF{}'.format(primary_source_id, j+1)] / pixel_scale 
					yoff = -head['S{}YOFF{}'.format(primary_source_id, j+1)] / pixel_scale
					raw_centers = np.array(image.shape) / 2.0

					# Append calculated values. 
					images.append(image)
					pas.append(rollang)
					centers.append([raw_centers[0]+xoff, raw_centers[1]+yoff])
					filenames.append(filename_prefix+'{}_{}'.format(i,j))
				else:
					#This is an unocculted off axis image. 
					if extract_offaxis == True and offaxis_image.size == 0:
						#Doesn't matter too much which specific image we use, just need an example. 
						# NOTE: Y offset must have its sign reversed due to differences between Pandeia scene construction and analysis of the image arrays
						offaxis_image = image
						offaxis_xoff = head['S{}OFFAX'.format(primary_source_id)]/pixel_scale
						offaxis_yoff = -head['S{}OFFAY'.format(primary_source_id)]/pixel_scale
						offaxis_target_center = [raw_centers[0]+offaxis_xoff, raw_centers[1]+offaxis_yoff]

						# Fit a 2D Gaussian to our offaxis source. 
						offaxis_image_smooth = pyklip.klip.nan_gaussian_filter(offaxis_image, lambda_d_pixel/2.355)
						offaxis_bestfit = fakes.gaussfit2d(offaxis_image_smooth, offaxis_target_center[0], offaxis_target_center[1], searchrad=4, guessfwhm=lambda_d_pixel, guesspeak=np.max(offaxis_image_smooth), refinefit=True)

						# Identify the peak flux and centroid of the Gaussian fit
						offaxis_peak_flux = offaxis_bestfit[0]
						offaxis_psf_xcen, offaxis_psf_ycen = offaxis_bestfit[2:4]

						# Extract a 20x20 pixel PSF stamp image using the x and y centroid. 
						x, y = np.meshgrid(np.arange(-20, 20.1, 1), np.arange(-20, 20.1, 1)) #Choice of 20x20 pixels as it encapsulates the bulk of the PSF
						x += offaxis_psf_xcen
						y += offaxis_psf_ycen
						offaxis_psf_stamp = scipy.ndimage.map_coordinates(offaxis_image, [y, x])
	
	##### Return a dictionary of the extracted images, their pas, centers, filenames, and some offaxis PSF information. 
	extracted = {'images':np.array(images), 'pas':np.array(pas), 'centers':np.array(centers), 'filenames':filenames, 'offaxis_psf_stamp':offaxis_psf_stamp, 'offaxis_peak_flux':offaxis_peak_flux}

	return extracted
 
def process_simulations(pancake_results, target, target_obs, filt, mask, primary_sources, references=None, reference_obs=None, target_rolls='default', reference_rolls='default', subtraction='ADI'):
	"""
	Function to process a set of desired simulated images from PanCAKE and convert them into pyKLIP datasets to enable easier stellar PSF subtraction
	and contrast curve estimation. 

	Parameters
		pancake_results : HDUList
			Simulated results as returned by pancake.sequence.Sequence().run()
		target : string
			The provided string name for the target scene in the observation sequence
		target_obs : list of stings
			List of target observation strings that correspond to extension names in the pancake_results HDUList
		filt : string
			JWST filter string
		mask : string
			JWST coronagraphic mask string
		primary_sources : list of strings
			List of the primary source(s) as obtained by identify_primary_sources
		references : list of strings
			The provided string name(s) for the reference scene(s) in the observations sequence, if any.
		reference_obs : list of strings
			List of target observation strings that correspond to extension names in the pancake_results HDUList, if any. 
		target_rolls : list of ints / floats
			Which target PA roll images to use. Alternatively, 'default' to use all of them for ADI modes, or roll=0 for RDI. 
		reference_rolls : list of ints / floats
			Which reference PA roll images to use. Alternatively, 'default' to use all of them for ADI modes, or roll=0 for RDI. 
		subtraction : str
			pyKLIP compatible subtraction string, available options are 'ADI', 'RDI', or 'ADI+RDI'
	Returns
		processed_output : dict
			Dictionary output containing pyKLIP datasets for the target and PSF library (if necessary), in addition
			to some information on the offaxis simulation for normalisation / planet injection purposes. 
	"""
	###### Get all of the observations names for this simulation
	obs_names = [pancake_results[i].header['EXTNAME'] for i in range(1,len(pancake_results))]

	##### Get the names for the target observations, and if necessary, get the reference observations too
	if not target_obs:
		raise ValueError("Unable to find specified target/filter/mask observation '{}/{}/{}' within simulated results. Possible observations include: {}".format(target, filt, mask, ', '.join(obs_names)))
	if references != None:
		reference_obs = [j for j in obs_names if any(ref in j for ref in references) and filt in j and mask in j]
		if not reference_obs:
			raise ValueError("Unable to find any reference/filter/mask observations of '{}/{}/{}' within simulated results. Possible observations include: {}".format("/{}/{}', '".format(filt, mask).join(references), filt, mask, ', '.join(obs_names)))

	##### Identify roll angles for the target and extract the data
	targ_available_rolls = list(dict.fromkeys([pancake_results[tob].header['ROLLANG'] for tob in target_obs]))
	if (target_rolls == 'default' and 'ADI' in subtraction) or target_rolls == 'all':
		# Default for the ADI scenario is to use all available rolls
		target_rolls = targ_available_rolls
	elif target_rolls == 'default' and 'ADI' not in subtraction:
		# Default for a non-ADI scenario is to use just the roll closest to 0 degrees
		target_rolls = [min([abs(j) for j in targ_available_rolls])]
		if len(targ_available_rolls) > 1:
			print('WARNING: No ADI requested, using the Roll={} simulation(s) for {}/{}/{}'.format(target_rolls[0], target, filt, mask))
	else:
		# User has specifically provide the rolls to use, check they exist. 
		for roll in target_rolls:
			if roll not in targ_available_rolls:
				raise ValueError("Unable to find any target observations at roll angle '{}' within simulated results. Possible roll angles include: {}".format(roll, ', '.join([str(t) for t in targ_available_rolls])))

	##### Need to access the saved simulation files and extract the necessary target data. 
	target_extracted = extract_simulated_images(pancake_results, target_obs, primary_sources, target_rolls, extract_offaxis=True, filename_prefix='target')
	
	##### If we are doing RDI at any point, also need to select the RDI rolls and extract the reference images. 
	if 'RDI' in subtraction:
		all_ref_available_rolls = []
		# Predefine an error message
		roll_err_mess = "Unable to find any reference observations at roll angle '{}' for Scene '{}' within simulated results. Possible roll angles include: {}"
		# Loop over input reference scenes 
		for reference in references:
			ref_available_rolls = list(dict.fromkeys([pancake_results[rob].header['ROLLANG'] for rob in reference_obs if ':'+reference+'_' in rob]))
			if reference_rolls in ['default', 'all']:
				#We will use all the available rolls
				all_ref_available_rolls.append(ref_available_rolls)
			elif isinstance(reference_rolls, (int, float)):
				roll = reference_rolls
				#There is only a single roll value provided, use if possible. 
				if roll not in ref_available_rolls: 
					raise ValueError(roll_err_mess.format(roll, reference, ', '.join([str(r) for r in ref_available_rolls])))
				all_ref_available_rolls.append([roll])
			elif isinstance(reference_rolls, list):
				#We have a list of rolls
				if not isinstance(reference_rolls[0], list):
					#Only one set of values, not nested lists, use for each reference if possible.
					for roll in reference_rolls:
						if roll not in ref_available_rolls: 
							raise ValueError(roll_err_mess.format(roll, reference, ', '.join([str(r) for r in ref_available_rolls])))
					all_ref_available_rolls.append(reference_rolls)
				elif len(reference_rolls) == len(references) and isinstance(reference_rolls[0], list):
					# We have a list of lists for each individual reference, use only the index corresponding to the individual reference. 
					rolls = reference_rolls[references.index(reference)]
					for roll in rolls:
						if roll not in ref_available_rolls: 
							raise ValueError(roll_err_mess.format(roll, reference, ', '.join([str(r) for r in ref_available_rolls])))
					all_ref_available_rolls.append(rolls)
			else:
				raise ValueError("Invalid format of reference rolls provided. Reference rolls must be provided either as an integer/float, a list of integer/floats, or a list of lists of integer/floats of equal length to the provided references.")
	
		##### Now, access the saved simulation files and extract the necessary reference data. 
		ref_extracted = extract_simulated_images(pancake_results, reference_obs, primary_sources, all_ref_available_rolls, references=references, filename_prefix='reference')

	##### Determine the 1 lambda / D inner working angle for this filter, and outer working angle based on image size
	pixel_scale = pancake_results[target_obs[0]].header['PIXSCALE']
	wavelength = float(re.findall('\\d+', filt)[0]) / 1e8
	lambda_d_arcsec = (wavelength / 6.5) * (180 / np.pi) * 3600
	lambda_d_pixel = lambda_d_arcsec / pixel_scale 
	
	inner_working_angle = 0#.5*lambda_d_pixel
	outer_working_angle = int(np.sqrt(2*(target_extracted['images'][0].shape[0]/2)**2))  #Go right to the corners


	##### Create the KLIP target dataset
	target_dataset = Instrument.GenericData(target_extracted['images'], target_extracted['centers'], IWA=inner_working_angle, parangs=target_extracted['pas'], filenames=target_extracted['filenames'])
	target_dataset.OWA = outer_working_angle

	##### Create the KLIP PSF library if necessary
	if 'RDI' in subtraction:
		psflib_data = np.append(ref_extracted['images'], target_extracted['images'], axis=0)
		psflib_filenames = np.append(ref_extracted['filenames'], target_extracted['filenames'], axis=0)
		psflib_centers = np.append(ref_extracted['centers'], target_extracted['centers'], axis=0)

		image_center = np.array(psflib_data[0].shape) / 2.0
		#Need to align the images so that they have the same centers. 
		for j, image in enumerate(psflib_data):
			recentered_image = pyklip.klip.align_and_scale(image, new_center=image_center, old_center=psflib_centers[j])
			psflib_data[j] = recentered_image

		psflib = rdi.PSFLibrary(psflib_data, image_center, psflib_filenames, compute_correlation=True)
		# Preparing of the PSF library can raise a future warning, ignore it to keep terminal clean. 
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', FutureWarning)
			psflib.prepare_library(target_dataset)
	else:
		# Return PSF library as NoneType if we aren't using a subtraction with RDI
		psflib = None

	##### Save the everything to a dictionary for easy access
	processed_output = {}
	processed_output['target_dataset'] = target_dataset
	processed_output['psflib'] = psflib
	processed_output['offaxis_psf_stamp'] = target_extracted['offaxis_psf_stamp']
	processed_output['offaxis_peak_flux'] = target_extracted['offaxis_peak_flux']

	return processed_output

def mask_companions(image_array, companion_xy, mask_radius=7):
	'''
	Function apply NaN masks to a number of images at the location of known companion objects. 

	Parameters
		image_array : 3D ndarray
			Numpy array of input images
		companion_xy : iterator of tuples
			zip() tuples, with each containing the companion x and y *pixel* locations.
		mask_radius : float
			The desired mask radius in pixels. 
	Returns
		masked_images : 3D ndarray
			Numpy array of output, companion masked images. 
	'''
	# Create an array to allocate the output images to. 	
	masked_images = np.empty_like(image_array)
	# Loop over images 
	for i, im in enumerate(image_array):
		# Create an index array 
		ydat, xdat = np.indices(im.shape)
		# Loop over companions, masking all pixels within a radius of the PSF FWHM
		for xy in companion_xy:
			distance_from_center = np.sqrt((xdat-xy[0])**2 + (ydat-xy[1])**2)
			comp_nan_mask = np.where(distance_from_center <= mask_radius)
			im[comp_nan_mask] = np.nan
		masked_images[i] = im

	return masked_images


def get_companion_mask(companion_xy, mask_dataset, mask_psflib, offaxis_psf_stamp, center=[0,0], filt='f444w', mask='mask335r', annuli=1, subsections=1, numbasis=25, movement=1, subtraction='ADI', outputdir='./RESULTS/'):
	'''
	Function to create a mask that can be applied to an image in order to "block" any pixels that correspond to the emitted flux of a companion object. 
	In essence, the function uses an offaxis PSF to inject companions into the image, on top of where they already exist, except at a *very* high flux. 
	This image can then be processed via KLIP to identify the pixels in the resultant subtracted image which are most impacted by the presence of 
	the companion object, and assign them to be masked. This offers significant improvements over a simplistic circular mask due to the lobes of the 
	JWST PSF, particularly for the NIRCam filters. Main current drawback is it only masks excess flux from companions, and misses ADI oversubtractions.  

	'''
	# Create our arrays of planets to inject into the input dataset, note the flux is scaled up by a factor of 10^12
	planet_inject = np.repeat(np.array([offaxis_psf_stamp])*1e6, mask_dataset.input.shape[0], axis=0)
	
	# Loop over each companion, and inject it into the image. 
	for xy in companion_xy:
		true_x = xy[0]-center[0]
		true_y = xy[1]-center[1]
		sep = np.sqrt(true_x**2 + true_y**2)
		pa = (np.arctan2(true_y, true_x) * 180 / np.pi)
		input_theta = pa % 360 #Let's keep everything positive, just in case
		fakes.inject_planet(frames=mask_dataset.input, centers=mask_dataset.centers, inputflux=planet_inject, astr_hdrs=mask_dataset.wcs, radius=sep, pa=None, thetas=[input_theta], field_dependent_correction=partial(transmission_corrected, filt=filt,  mask=mask))

	# Perform KLIP on the new images. 
	fileprefix = "FOR_MASKING" #Adjustable if necessary
	filesuffix = "-KLmodes-all.fits" #Don't adjust
	parallelized.klip_dataset(mask_dataset, outputdir=outputdir, fileprefix=fileprefix, annuli=annuli, subsections=subsections, numbasis=numbasis, mode=subtraction, psf_library=mask_psflib, movement=movement, verbose=False)

	# Open the KLIP file and read back in the subtracted image
	injected_file = "{}/{}{}".format(outputdir, fileprefix, filesuffix)
	with fits.open(injected_file) as hdulist:
		raw_injected_image = hdulist[0].data[0]
		image_x, image_y  = raw_injected_image.shape
		image_dx = np.tile(np.arange(-(image_x-1)/2, (image_x)/2), (image_y, 1))
		image_dy = np.tile(np.arange(-(image_y-1)/2, (image_y)/2), (image_x, 1)).transpose()
		injected_image_centers = [hdulist[0].header["PSFCENTX"], hdulist[0].header["PSFCENTY"]]
		injected_image = raw_injected_image#transmission_corrected(raw_injected_image, image_dx, image_dy, filt, mask, mode='divide') #Correct for coronagraph transmission again

	# Assign a mask for pixels above a certain flux threshold. 5e-2 times the max seems to do a good job for all filters, fine tuning may
	# improve things slightly, but likely not for all filters. 
	comp_mask = np.where(injected_image > 5e-2*np.nanmax(injected_image))

	return comp_mask


def meas_contrast_basic(dat, iwa, owa, resolution, center=None, low_pass_filter=True):
    """
	Duplicate of the meas_contrast funciton within pyKLIP, except calculating a 
	standard 5 sigma limit instead of small sample statistics corrections. 

    """

    if center is None:
        starx = dat.shape[1]//2
        stary = dat.shape[0]//2
    else:
        starx, stary = center

    # figure out how finely to sample the radial profile
    dr = resolution/2.0
    numseps = int((owa-iwa)/dr)
    # don't want to start right at the edge of the occulting mask
    # but also want to well sample the contrast curve so go at twice the resolution
    seps = np.arange(numseps) * dr + iwa + resolution/2.0
    dsep = resolution
    # find equivalent Gaussian PSF for this resolution


    # run a low pass filter on the data, check if input is boolean or a number
    if not isinstance(low_pass_filter, bool):
        # manually passed in low pass filter size
        sigma = low_pass_filter
        filtered = pyklip.klip.nan_gaussian_filter(dat, sigma)
    elif low_pass_filter:
        # set low pass filter size to be same as resolution element
        sigma = dsep / 2.355  # assume resolution element size corresponds to FWHM
        filtered = pyklip.klip. nan_gaussian_filter(dat, sigma)
    else:
        # no filtering
        filtered = dat

    contrast = []
    # create a coordinate grid
    x,y = np.meshgrid(np.arange(float(dat.shape[1])), np.arange(float(dat.shape[0])))
    r = np.sqrt((x-starx)**2 + (y-stary)**2)
    theta = np.arctan2(y-stary, x-starx) % 2*np.pi
    for sep in seps:
        # calculate noise in an annulus with width of the resolution element
        annulus = np.where((r < sep + resolution/2) & (r > sep - resolution/2))
        noise_mean = np.nanmean(filtered[annulus])
        noise_std = 5*np.nanstd(filtered[annulus], ddof=1)

    return seps, np.array(noise_std)

def compute_contrast(subtracted_hdu_file, filt, mask, offaxis_psf_stamp, offaxis_flux, raw_input_dataset, raw_input_psflib, primary_vegamag=0, pixel_scale=0.063, annuli=1, subsections=1, numbasis=25, movement=1, subtraction='ADI', companion_xy=None, verbose=True, outputdir='./RESULTS/', plot_klip_throughput=False):
	"""
	Function to compute contrast curves from a pyKLIP subtracted image file. Contrast curves will be corrected for both the coronagraphic and KLIP throughput, 
	in addition to being converted to relative, and absolute magnitude sensitivity limits. 

	Parameters
		subtracted_hdu_file : str
			Filename for a subtracted image file as produced by pyklip.parallelized.klip_dataset()
		filt : str
			JWST filter string
		mask : str
			JWST coronagraphic mask string
		offaxis_psf_stamp : 2D ndarray
			Stamp image of an offaxis (i.e. not underneath the coronagraph) PSF
		offaxis_flux : float
			Peak flux of the offaxis PSF
		raw_input_dataset : pyKLIP Dataset
			The input target dataset that was used to generate the subtracted_hdu_file. Used many times over for planet injection. 
		raw_input_psflib : pyKLIP PSFLibrary
			The input PSF library dataset that was used to generate the subtracted_hdu_file, if any. Used many times over for planet injection. 
		primary_vegamag : float
			Vega magnitude of the primary source of the target scene in the specified filter.
		pixel_scale : float
			Pixel scale for this observation. 
		annuli : int
			pyKLIP argument - Annuli to use for KLIP. Can be a number, or a list of 2-element tuples (a, b) specifying the pixel 
			boundaries (a <= r < b) for each annulus
		subsections : int 
			pyKLIP argument - Sections to break each annuli into. Can be a number [integer], or a list of 2-element tuples (a, b) 
			specifying the positon angle boundaries (a <= PA < b) for each section [radians]
		numbasis : int
			number of KL basis vectors to use (can be a scalar or list like). Length of b If numbasis is [None] the number of KL modes to be 
			used is automatically picked based on the eigenvalues.
		movement : int
			pyKLIP argument - minimum amount of movement (in pixels) of an astrophysical source to consider using that image for a refernece PSF
		subtraction : str
			pyKLIP compatible subtraction string, available options are 'ADI', 'RDI', or 'ADI+RDI' 
		companion_xy : iterator of tuples
			zip() tuples, with each containing any companion x and y *pixel* locations.
		verbose : bool
			Optional argument to turn on (True) or off (False) printed terminal updates. 
		outputdir : str
			Directory to save any temporary or results files. 
		plot_klip_throughput : bool
			Optional argument to plot (True) the calculated KLIP throughput for each image. Primarily for debugging.  
	Returns 
		all_contrasts : dict
			Dictionary output of the contrast, relative magnitude sensitivity, absolute magnitude sensitivity and separation. Alternative
			formats of the separation, the contrast prior to throughput corrections, and the estimated KLIP throughput, are also provided. 
	"""

	##### First thing we need to do is open the file with the subtracted images. 
	with fits.open(subtracted_hdu_file) as hdulist:
		subtracted_hdu = hdulist[0]
		raw_image = subtracted_hdu.data[0]
		center = [subtracted_hdu.header["PSFCENTX"], subtracted_hdu.header["PSFCENTY"]]

	##### Get initial properties based on filter and mask input.	
	wavelength = float(re.findall('\\d+', filt)[0]) / 1e8
	lambda_d_arcsec = (wavelength / 6.5) * (180 / np.pi) * 3600
	lambda_d_pixel = lambda_d_arcsec / pixel_scale 
	
	# Set the IWA to 0.5 lambda/D, contrast will be calculated from 1 lambda / D
	inner_working_angle = 0.5*lambda_d_pixel 
	# For the OWA, get inaccurate measurements close to the edges of the simulated image, so only go 95% of the way. 
	# In reality many more pixels will be available as simulation is a small portion of detector, so this effect is artificial.
	outer_working_angle = 0.95*int(raw_image.shape[0]/2) 

	##### Normalise by the coronagraphic throughput
	image_x, image_y  = raw_image.shape
	image_dx = np.tile(np.arange(-(image_x-1)/2, (image_x)/2), (image_y, 1))
	image_dy = np.tile(np.arange(-(image_y-1)/2, (image_y)/2), (image_x, 1)).transpose()
	image = transmission_corrected(raw_image, image_dx, image_dy, filt, mask, mode='divide')

	##### If we aren't using a spherically symmetric coronagraph, we should mask out certain pixels to improve the contrast measurement. 
	if mask in ['MASKSWB', 'MASKLWB', 'FQPM1065', 'FQPM1140', 'FQPM1550']:
		mask_throughput = raw_image / image
		# Identify a level at which pixels should be set to nans, be slightly more aggresive for MIRI
		if mask in ['MASKSWB', 'MASKLWB']: 
			nan_cut = 0.5
		elif mask in ['FQPM1065', 'FQPM1140', 'FQPM1550']: 
			nan_cut = 0.65

		nan_mask = np.where(mask_throughput<nan_cut)
		raw_image[nan_mask] = np.nan
		image[nan_mask] = np.nan

	##### Also, if there are companions in the image, these should be masked out. 
	if companion_xy != None: 
		comp_mask_dataset = deepcopy(raw_input_dataset)
		comp_mask_psflib = deepcopy(raw_input_psflib)
		comp_mask = get_companion_mask(companion_xy, comp_mask_dataset, comp_mask_psflib, offaxis_psf_stamp, center=center, filt=filt, mask=mask, annuli=annuli, subsections=subsections, numbasis=numbasis, subtraction=subtraction, movement=movement, outputdir=outputdir)
		raw_image[comp_mask] = np.nan
		image[comp_mask] = np.nan

	##### Divide by the peak of the offaxis flux to convert the images into contrast
	raw_image /= offaxis_flux
	image /= offaxis_flux

	##### Grab a contrast measurement before and after the coronagraphic throughput correction
	uncorr_contrast_seps, uncorr_contrast = pyklip.klip.meas_contrast(dat=raw_image, iwa=inner_working_angle, owa=outer_working_angle, resolution=lambda_d_pixel, center=center, low_pass_filter=lambda_d_pixel/2.355)
	contrast_seps, contrast = pyklip.klip.meas_contrast(dat=image, iwa=inner_working_angle, owa=outer_working_angle, resolution=lambda_d_pixel, center=center, low_pass_filter=lambda_d_pixel/2.355)

	uncorr_contrast_seps_raw5sig, uncorr_contrast_raw5sig = meas_contrast_basic(dat=raw_image, iwa=inner_working_angle, owa=outer_working_angle, resolution=1, center=center, low_pass_filter=lambda_d_pixel/2.355)
	contrast_seps_raw5sig, contrast_raw5sig = meas_contrast_basic(dat=image, iwa=inner_working_angle, owa=outer_working_angle, resolution=1, center=center, low_pass_filter=lambda_d_pixel/2.355)

	##### At this stage contrast is usable, but has not been calibrated for the KLIP throughput. 
	##### Need to inject planets into the raw_image and see how well they are recovered 
	#Define injection values 
	min_sep = inner_working_angle # Minimum injection separation
	max_sep = outer_working_angle # Maximum injection separation
	
	nplanets = 1   #Just inject 1 planet at a time, but can do multiple at once. 
	inter_planet_sep = 2 # Radial separation in pixels between each planet injection.
	start_pas = np.linspace(0, 360*(nplanets-1)/nplanets, nplanets) #Starting PA's for the injected planets. 
	num_pas = 5 # Number of different PA's to look at for each separation. Should avoid values of 1,2,4 for MIRI as they will lie behind the 4QPM edges. 

	# Maximum separation of first iteration and number of loops needed. 
	max_sep_1 = min_sep + (inter_planet_sep * (nplanets-1))
	n_sep_loops = int((((max_sep - min_sep)/(inter_planet_sep)) + 1)/nplanets)

	#Offaxis psf stamp to use for injection
	psf_stamp_input = np.array([offaxis_psf_stamp]) 

	retrieved_fluxes_all, planet_pas_all, planet_seps_all, input_contrasts_all = [], [], [], []
	if verbose:	print('--> Determining KLIP Throughput')
	for n in range(n_sep_loops):
		# Create array of separations and contrasts to be injected at, spaced by desired distance b/t planets
		planet_seps = np.arange(min_sep + (n*nplanets*inter_planet_sep), max_sep_1+1 + (n*nplanets*inter_planet_sep), inter_planet_sep)

		# Gather contrasts at the separation of the injected planet for the *uncorrected* images, i.e coronagraph throughput is still there. 
		input_contrasts = (np.interp(planet_seps, contrast_seps, uncorr_contrast))*4000

		# Loop over however many PA's were requested. 
		for num_pa in range(num_pas):
			# Copy our dataset, otherwise we'll keep injecting planets into the same image. 
			input_dataset = deepcopy(raw_input_dataset)
			input_psflib = deepcopy(raw_input_psflib)

			# Rotate the planets between each iteration, equally spaced based on num_pas - plus a slight deviation to break symmetries. 
			planet_pas = [x+(360*num_pa)/num_pas+1 for x in start_pas]
			
			# Loop over all of the planets to be injected. 
			injected = [] # Record whether planets were injected or not
			for input_contrast, sep, pa in zip(input_contrasts, planet_seps, planet_pas):
				# Let's do some checks to make sure we want to inject a planet here
				perform_injection = True

				# Make sure we don't inject planets on top of, or close to, existing planets
				if companion_xy != None:
					#Get pixel position, make sure to minus the x position as we are going counterclockwise
					check_pos_x = -(sep * np.sin(pa*np.pi/180)) + center[0]
					check_pos_y = (sep * np.cos(pa*np.pi/180)) + center[1]
					for xy in companion_xy:
						dist_planet = np.sqrt((check_pos_x - xy[0])**2 + (check_pos_y - (image.shape[1] - xy[1]) )**2)
						# Don't inject planets within 2 lambda/D of an existing companion. 
						if dist_planet < 2*lambda_d_pixel: 
							perform_injection = False

				if perform_injection:
					base_planet_flux = psf_stamp_input * input_contrast
					planet_flux = np.repeat(base_planet_flux, input_dataset.input.shape[0], axis=0) #Need to pass as many planet PSF stamps as there are target images.  
					### Tidy later, but essentially if specifying the input thetas, you have to do so for every input ADI frame, not just a single angle. 
					### Also, for some reason, the array needs to be flipped relative to the input_dataset.PAs to line things up correctly. 
					input_thetas = ((270 - pa - input_dataset.PAs) % 360)
					fakes.inject_planet(frames=input_dataset.input, centers=input_dataset.centers, inputflux=planet_flux, astr_hdrs=input_dataset.wcs, radius=sep, pa=None, thetas=input_thetas, field_dependent_correction=partial(transmission_corrected, filt=filt,  mask=mask))
					injected.append(True)
				else:
					# Just move to the next injection location
					injected.append(False)
					continue

			# Now want to run KLIP on these planet injected images
			fileprefix = "FAKE_INJECTED_{}".format(num_pas) #Adjustable if necessary
			filesuffix = "-KLmodes-all.fits" #Don't adjust
			parallelized.klip_dataset(input_dataset, outputdir=outputdir, fileprefix=fileprefix, annuli=annuli, subsections=subsections, numbasis=numbasis, mode=subtraction, psf_library=input_psflib, movement=movement, verbose=False)
	
			#Reopen produced file from pyKLIP
			injected_file = "{}{}{}".format(outputdir, fileprefix, filesuffix)
			with fits.open(injected_file) as hdulist:
				raw_injected_image = hdulist[0].data[0]
				injected_image_centers = [hdulist[0].header["PSFCENTX"], hdulist[0].header["PSFCENTY"]]
				injected_image = transmission_corrected(raw_injected_image, image_dx, image_dy, filt, mask, mode='divide') #Correct for coronagraph transmission again

			# Retrieve the planetary flux and append it to our initial array
			retrieved_planet_fluxes, used_contrasts, used_seps, used_pas = [], [], [], []
			for input_contrast, sep, pa, inj in zip(input_contrasts, planet_seps, planet_pas, injected):
				# Planet injection step.
				if inj == True:
					# Before we retrieve the flux, we need to apply a low pass filter (smoothing) to the images, as this is what was done to 
					# obtain the contrast curve. Make sure to use the same value of lambda_d_pixel/2.355. 
	
					injected_image = pyklip.klip.nan_gaussian_filter(injected_image, lambda_d_pixel/2.355)
					input_theta = (270 - pa) % 360
					fake_flux = fakes.retrieve_planet_flux(frames=injected_image, centers=injected_image_centers, astr_hdrs=input_dataset.output_wcs[0], sep=sep, pa=None, thetas=input_theta, searchrad=int(lambda_d_pixel*2), guessfwhm=lambda_d_pixel)

					retrieved_planet_fluxes.append(fake_flux)
					used_contrasts.append(input_contrast)
					used_seps.append(sep)
					used_pas.append(pa)

			# Add things to the arrays defined at the start
			retrieved_fluxes_all.extend(retrieved_planet_fluxes)
			planet_pas_all.extend(used_pas)
			planet_seps_all.extend(used_seps)
			input_contrasts_all.extend(used_contrasts)

			# Delete the injected planet image, as it will only clutter the directory
			os.remove(injected_file)

	##### So, now planets have been injected into a variety of images and the flux prior to and after the injection has been measured. 
	##### What we can now do is use this flux ratio to calibrate the contrast curve for the KLIP throughput. 

	# Create a table of all variables
	inject_vars = Table([retrieved_fluxes_all, planet_seps_all, input_contrasts_all, planet_pas_all], names=("flux", "separation", "input_contrast", "pas"))
	
	inject_vars["input_flux"] = inject_vars["input_contrast"] * offaxis_flux #Peak flux from the offaxis PSF 
	inject_vars["throughput"] = inject_vars["flux"] / inject_vars["input_flux"] # Calculate throughput and add it to the table

	inject_vars_grouped = inject_vars.group_by("separation") # Group by separation
	med_inject_vars = inject_vars_grouped.groups.aggregate(np.nanmedian) # Calculate the median values at each separation

	# Can't have a negative throughput, but can arise erroneously very close to the primary source due to speckle noise.
	# Counteract this by setting all negative throughput values to a very low value of 1e-10
	med_inject_vars["throughput"][np.where(med_inject_vars['throughput']<0)] = 1e-10

	# Find the throughput for the separations the contrast curve has been computed at
	throughput_interp = np.interp(contrast_seps, med_inject_vars['separation'], med_inject_vars["throughput"])

	# Apply median KLIP throughput to the 5 sigma contrast curve which **has** been already corrected for the coronagraph throughput 
	klip_corrected_contrast = contrast / throughput_interp

	##### Convert to relative / absolute magnitudes.
	relmag = -2.5*np.log10(klip_corrected_contrast)
	absmag = relmag + primary_vegamag

	##### Now make it all again, but for the basic 5 sigma curves with no small separation correction
	throughput_interp2 = np.interp(contrast_seps_raw5sig, med_inject_vars['separation'], med_inject_vars['throughput'])
	klip_corrected_contrast_raw5sig = contrast_raw5sig / throughput_interp2
	relmag_raw5sig = -2.5*np.log10(klip_corrected_contrast_raw5sig)
	absmag_raw5sig = relmag_raw5sig + primary_vegamag

	# Prepare dictionary to return contrasts, give things more descriptive names for users. 
	all_contrasts = {'separation':contrast_seps, 'separation_arcsec':contrast_seps*pixel_scale, 'separation_lambdad':contrast_seps/lambda_d_pixel, 'contrast_noklipthrput_nocorothrput':uncorr_contrast, 'contrast_noklipthrput':contrast, 'contrast':klip_corrected_contrast, 'klipthrput':throughput_interp, 'relmag':relmag, 'absmag':absmag, 'separation_raw5sig':contrast_seps_raw5sig, 'separation_arcsec_raw5sig':contrast_seps_raw5sig*pixel_scale, 'separation_lambdad_raw5sig':contrast_seps_raw5sig/lambda_d_pixel,'contrast_noklipthrput_nocorothrput_raw5sig':uncorr_contrast_raw5sig, 'contrast_noklipthrput_raw5sig':contrast_raw5sig, 'contrast_raw5sig':klip_corrected_contrast_raw5sig, 'relmag_raw5sig':relmag_raw5sig, 'absmag_raw5sig':absmag_raw5sig}

	if plot_klip_throughput:
		plt.figure()
		ax = plt.gca()
		ax.plot(med_inject_vars["separation"], med_inject_vars["throughput"], color="#577B51", label="Median Throughput")
		ax.scatter(inject_vars["separation"], inject_vars["throughput"], color = '#95B2B8', alpha=0.5, edgecolors='black', s=80)
		ax.set_ylim(0,1)
		ax.set_ylabel("Throughput")
		ax.set_xlabel("Planet Separation")
		ax.set_title("Algorithm Throughput")
		ax.legend(frameon=False, loc="lower left")
		plt.show()

	return all_contrasts

def get_source_properties(template_obs, primary_source):
	header = template_obs.header
	pixel_scale = header['PIXSCALE']
	num_sources = header['NSOURCES']

	raw_center = np.array(template_obs.data[0].shape) / 2.0

	sources = [header['SOURCE{}'.format(j+1)] for j in range(num_sources)]
	primary_source_id = sources.index(primary_source)

	# Gather target props
	target_primary_vegamag = template_obs.header['S{}VGAMG'.format(primary_source_id+1)]
	target_xoff = template_obs.header['S{}XOFF1'.format(primary_source_id+1)] / pixel_scale
	target_yoff = -template_obs.header['S{}YOFF1'.format(primary_source_id+1)] / pixel_scale

	if num_sources > 1:
		# Gather companion props
		comp_xoffs = np.array([header['S{}XOFF1'.format(j+1)]/pixel_scale for j in range(num_sources) if j != primary_source_id])
		comp_yoffs = np.array([-header['S{}YOFF1'.format(j+1)]/pixel_scale for j in range(num_sources) if j != primary_source_id])	
		
		comp_seps = np.sqrt((comp_xoffs-target_xoff)**2 + (comp_yoffs-target_yoff)**2) * pixel_scale
		comp_xy = [[x,y] for x,y, in zip(comp_xoffs+raw_center[0], comp_yoffs+raw_center[1])]
		comp_names = [header['SOURCE{}'.format(j+1)] for j in range(num_sources) if j != primary_source_id]

		comp_vegamags = [header['S{}VGAMG'.format(j+1)] for j in range(num_sources) if j != primary_source_id]
		comp_contrasts = 10**(-0.4 * (np.array(comp_vegamags) - target_primary_vegamag))
	else:
		comp_vegamags, comp_seps, comp_xy, comp_names, comp_contrasts = None, None, None, None, None

	source_props = {'target_primary_vegamag':target_primary_vegamag, 'comp_vegamags':comp_vegamags, 'comp_seps':comp_seps, 'comp_xy':comp_xy, 'comp_names':comp_names, 'comp_contrasts':comp_contrasts}

	return source_props

def companion_snrs(subtracted_hdu_file, filt, mask, companion_xy, mask_radius=7):
	# Read in the file
	with fits.open(subtracted_hdu_file) as hdulist:
		subtracted_hdu = hdulist[0]
		raw_image = subtracted_hdu.data[0]
		center = [subtracted_hdu.header["PSFCENTX"], subtracted_hdu.header["PSFCENTY"]]

	# Correct for the coronagraphic throughput
	image_x, image_y  = raw_image.shape
	image_dx = np.tile(np.arange(-(image_x-1)/2, (image_x)/2), (image_y, 1))
	image_dy = np.tile(np.arange(-(image_y-1)/2, (image_y)/2), (image_x, 1)).transpose()
	image = transmission_corrected(raw_image, image_dx, image_dy, filt, mask, mode='divide')

	# Construct a grid with the same shape as the image, compute radial distance from image center
	x,y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
	rad_dist = np.sqrt((x-center[0])**2+(y-center[1])**2)

	# Mask the coronagraphic bar
	if mask in ['MASKSWB', 'MASKLWB', 'FQPM1065', 'FQPM1140', 'FQPM1550']:
		mask_throughput = raw_image / image

		# Identify a level at which pixels should be set to nans. Be less aggresive than contrast measurement.
		if mask in ['MASKSWB', 'MASKLWB']: 
			nan_cut = 0.5
		elif mask in ['FQPM1065', 'FQPM1140', 'FQPM1550']: 
			nan_cut = 0.7

		nan_mask = np.where(mask_throughput<nan_cut)
		image[nan_mask] = np.nan

	# Mask companions 
	masked_image = mask_companions([deepcopy(image)], companion_xy, mask_radius=mask_radius)[0]

	##### Compute the 2D SNR
	annulus_width = 1
	# Build a lambda function to compute the standard deviation within an annulus
	f = lambda r : np.nanstd(masked_image[(rad_dist >= r-annulus_width/2) & (rad_dist < r+annulus_width/2)])
	# Define the radial separations 
	r  = np.linspace(0,100,num=100)
	# Catch RuntimeWarnings from the np.nanstd() function at separations where all pixels == NaN
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', RuntimeWarning)
		# Pass separations to our function, vectorize so it accepts numpy arrays
		std = np.vectorize(f)(r) 
	std_interp = interp1d(r, std) # Interpolate the results
	# Use radial array as a template array 
	std_2d = deepcopy(rad_dist)
	for i, row in enumerate(std_2d):
		for j, item in enumerate(row):
			std_2d[i,j] = std_interp(item) #Assign each pixel it's standard deviation using the interpolation

	# Divide through to obtain the 2D SNR map.
	image_snr = image / std_2d
	
	##### Now for each companion, try to fit a 2D gaussian to its location to estimate the peak SNR
	companion_snrs = []
	for comp in companion_xy:
		bestfit = fakes.gaussfit2d(image_snr, comp[0], comp[1], searchrad=3, guessfwhm=2, refinefit=True)
		companion_snrs.append(bestfit[0])

	return companion_snrs

def contrast_curve(pancake_results, target, references=None, subtraction='ADI', filters='all', masks='all', target_rolls='default', target_primary_source='default', reference_primary_sources='default', reference_rolls='default', klip_annuli=1, klip_subsections=1, klip_numbasis=25, klip_movement=1, get_companion_snrs=True, clean_saved_files=False, outputdir='./RESULTS/', save_prefix='default', verbose=True, plot_contrast=True, plot_klip_throughput=False, save_contrasts=True):

	###### Perform some input checks on the user provided parameters
	# Check requested subtraction is valid
	sub_methods = ['ADI', 'RDI', 'RDI+ADI', 'ADI+RDI']
	if subtraction not in sub_methods:
		raise ValueError("Specified subtraction method '{}' not valid, possible methods are: {}".format(subtraction, ', '.join(sub_methods)))
	if subtraction in ['RDI', 'RDI+ADI', 'ADI+RDI'] and references == None:
		raise ValueError("Must identify reference scene observations through the 'references' argument.")
	if subtraction == 'ADI' and references != None:
		if verbose: print('Specified reference scenes will not be used for ADI subtraction!')
		references = None

	# Check the output directory exists, if not then create it
	if not os.path.exists(outputdir):
		os.makedirs(outputdir)

	#Convert certain arguments to lists if necessary
	if isinstance(references, str): references = [references]
	if isinstance(target_rolls, (int, float)): target_rolls = [target_rolls] 
	if isinstance(klip_numbasis, int): klip_numbasis = [klip_numbasis]
	if isinstance(filters, str): filters = [filters] 
	if isinstance(masks, str): masks = [masks] 

	# Make sure strings are upper case
	if filters[0] != 'all': filters = [filt.upper() for filt in filters]
	if masks[0] != 'all': masks = [mask.upper() for mask in masks]

	#Get all of the observation / scenenames
	obs_names = [pancake_results[i].header['EXTNAME'] for i in range(1,len(pancake_results))]

	#Assign filters if no specifics are requested. 
	if filters == ['all']:
		filters = list(dict.fromkeys([i.split('_')[-1] for i in obs_names if ':'+target+'_' in i]))

	##### Identify the Sources that correspond to the central "star" for all of the used scenes. 
	primary_sources = identify_primary_sources(pancake_results, target, references=references, target_primary_source=target_primary_source, reference_primary_sources=reference_primary_sources)

	##### Start loop for creating contrast curves
	if verbose:	print('Computing Contrast Curves ({})...'.format(subtraction))
	contrast_curve_dict = {} #This is where all the contrast curves will be saved
	for filt in filters:
		### Get all of the target observations based on filter 
		raw_target_obs = [j for j in obs_names if target in j and filt in j]
		if not raw_target_obs:
			raise ValueError("Unable to find specified target/filter observation '{}/{}' within simulated results. Possible observations include: {}".format(target, filt, ', '.join(obs_names)))
			
		all_simulated_masks = list(dict.fromkeys([j.split('_')[-2] for j in raw_target_obs])) #This is all of the masks simulated for this filter

		###If all masks were requested, just use the all that are available. If not, use those specifically requested 
		if masks[0] == 'all': 
			used_masks = all_simulated_masks
		else:
			used_masks = masks

		### Now loop over all of the masks
		for mask in used_masks:
			print('{} // {}+{}'.format(target, filt.upper(), mask.upper()))

			### Get the target observations just for this mask
			target_obs = [j for j in raw_target_obs if mask in j]

			if not target_obs:
				raise ValueError("Unable to find specified target/mask/filter observation '{}/{}/{}' within simulated results. Possible observations include: {}".format(target, mask, filt, ', '.join(obs_names)))
			
			### If necessary, get the reference observations
			if references != None:
				reference_obs = [j for j in obs_names if any(ref in j for ref in references) and filt in j and mask in j]
				if not reference_obs:
					raise ValueError("Unable to find any reference/mask/filter observations of '{}/{}/{}' within simulated results. Possible observations include: {}".format("/{}/{}', '".format(mask, filt).join(references), filt, mask, ', '.join(obs_names)))
			else:
				reference_obs = None

			##### Create the KLIP datasets
			processed = process_simulations(pancake_results, target, target_obs, filt, mask, primary_sources, references=references, reference_obs=reference_obs, target_rolls=target_rolls, reference_rolls=reference_rolls, subtraction=subtraction)

			##### Perform the subtraction
			### Prior to the subtraction, must duplicate the datasets for KLIP throughput calculations
			target_dataset_throughput = deepcopy(processed['target_dataset'])
			if processed['psflib'] != None:
				psflib_throughput = deepcopy(processed['psflib'])
				# Preparing of the PSF library can raise a future warning, ignore it to keep terminal clean. 
				with warnings.catch_warnings():
					warnings.simplefilter('ignore', FutureWarning)
					psflib_throughput.prepare_library(target_dataset_throughput)
			else:
				psflib_throughput = None

			### Define the prefix to save files
			if save_prefix == 'default':
				klip_nb_str = ','.join([str(j) for j in klip_numbasis])
				true_save_prefix = "{}-{}-{}-{}-nb{}a{}s{}m{}".format(target, filt, mask, subtraction, klip_nb_str, klip_annuli, klip_subsections, klip_movement)
			else:
				true_save_prefix = save_prefix 

			### Subtraction routine
			if verbose:	print('--> Performing KLIP Subtraction')
			parallelized.klip_dataset(processed['target_dataset'], outputdir=outputdir, fileprefix=true_save_prefix, annuli=klip_annuli, subsections=klip_subsections, numbasis=klip_numbasis, mode=subtraction, psf_library=processed['psflib'], movement=klip_movement, verbose=False)
			#This function doesn't return anything. Instead, all information is saved to a FITS file.  
			subtracted_hdu_file = outputdir + "{}-KLmodes-all.fits".format(true_save_prefix)

			# Get some information on the sources in our scene. 
			source_props = get_source_properties(pancake_results[target_obs[0]], primary_sources[0])

			##### Now want to turn the image results into a contrast curve for this mask and filter combination
			# Quickly grab the pixel scale for this filter 
			pixel_scale = pancake_results[target_obs[0]].header['PIXSCALE'] 
			# Also need the vegamag of the primary source in the target image for this specific filter
			nsources = pancake_results[target_obs[0]].header['NSOURCES']

			if verbose:	print('--> Extracting Contrast Curve')
			all_contrasts = compute_contrast(subtracted_hdu_file, filt, mask, processed['offaxis_psf_stamp'], processed['offaxis_peak_flux'], target_dataset_throughput, psflib_throughput, primary_vegamag=source_props['target_primary_vegamag'], pixel_scale=pixel_scale, annuli=klip_annuli, subsections=klip_subsections, numbasis=klip_numbasis, subtraction=subtraction, movement=klip_movement, companion_xy=source_props['comp_xy'], outputdir=outputdir, plot_klip_throughput=plot_klip_throughput)

			contrast_curve_dict['{}+{}'.format(filt.upper(), mask.upper())] = all_contrasts

			##### Now that we've done all this, it's also possible to grab the SNR for any companions in the image
			if get_companion_snrs:
				if source_props['comp_xy'] == None:
					if verbose:	print('WARNING: Unable to compute companion SNR as no companions were identified in target image.')
				else:
					if verbose:	print('--> Estimating Companion SNR')
					snrs = companion_snrs(subtracted_hdu_file, filt, mask, source_props['comp_xy'])
					contrast_curve_dict['{}+{}'.format(filt.upper(), mask.upper())]['companion_snrs'] = snrs
					contrast_curve_dict['{}+{}'.format(filt.upper(), mask.upper())]['companion_contrast'] = source_props['comp_contrasts']
					contrast_curve_dict['{}+{}'.format(filt.upper(), mask.upper())]['companion_seps'] = source_props['comp_seps']
					contrast_curve_dict['{}+{}'.format(filt.upper(), mask.upper())]['companion_names'] = source_props['comp_names']

			# Save output contrasts to a file so things don't need to be calculated again.
			if save_contrasts:
				# Create file name
				ccurve_save_file = subtracted_hdu_file.replace('.fits', '_CURVES.json')
				# Create a function to convert numpy arrays to lists within the dictionary
				def default(obj):
					if isinstance(obj, np.ndarray):
						return obj.tolist()
					raise TypeError('Value in contrast curve dictionary not serializable by JSON.')

				# Save file
				with open(ccurve_save_file, 'w') as f:
					json.dump(contrast_curve_dict[filt.upper()+'+'+mask.upper()], f, sort_keys=True, indent=4, default=default)

			# Default behaviour is to leave KLIP subtracted files, but if requested the files **for this run only** will be deleted.  
			if clean_saved_files:
				os.remove(subtracted_hdu_file)

			# Plot contrast curve for this filter/mask combo if requested. 
			if plot_contrast:
				##### First make a plot for the straightforward contrast 
				plot_save_file = subtracted_hdu_file.replace('.fits', '_CURVES.png')
				plt.figure(figsize=(12,7))
				ax = plt.gca()
				separation = contrast_curve_dict['{}+{}'.format(filt.upper(), mask.upper())]['separation_arcsec']
				ax.plot(separation, all_contrasts['contrast'], color="#577B51", linewidth = 3, label = '5$\\sigma$ Contrast')
				ax.plot(all_contrasts['separation_arcsec_raw5sig'], all_contrasts['contrast_raw5sig'], color="#A1BF9C", linewidth = 3, label = '5$\\sigma$ Standard Deviation', ls=':')
				# Also add companion magnitudes if necessary. 
				if source_props['comp_seps'] != None and source_props['comp_contrasts']:
					ax.scatter(source_props['comp_seps'], source_props['comp_contrasts'], c='w', edgecolors='k', linewidths=2, s=50)
					for j, name in enumerate(source_props['comp_names']):
						ax.annotate(name, (source_props['comp_seps'][j], source_props['comp_contrasts'][j]), xytext=(5, 5), textcoords='offset points')
				ax.legend(frameon=False, fontsize=14)
				ax.set_yscale('log')
				ax.set_ylabel("Contrast", fontsize=16)
				ax.set_xlabel('Separation (")', fontsize=16)
				ax.tick_params(which='both', direction='in', labelsize=14, axis='both', top=True, right=True)
				ax.set_title('{} // {}+{}'.format(target, filt.upper(), mask.upper()), fontsize=18)
				plt.savefig(plot_save_file, bbox_inches='tight', dpi=300)

				##### Now make a plot for the absolute magnitude contrast limit reached. 
				plot_save_file = subtracted_hdu_file.replace('.fits', '_MAGCURVES.png')
				plt.figure(figsize=(12,7)) 
				ax = plt.gca()
				separation = contrast_curve_dict['{}+{}'.format(filt.upper(), mask.upper())]['separation_arcsec']
				ax.plot(separation, all_contrasts['absmag'], color="#577B51", linewidth = 3, label = '5$\\sigma$ Sensitivity Limit')
				ax.plot(all_contrasts['separation_arcsec_raw5sig'], all_contrasts['absmag_raw5sig'], color="#A1BF9C", linewidth = 3, label = '5$\\sigma$ Standard Deviation', ls=':')
				# Also add companion magnitudes if necessary. 
				if source_props['comp_seps'] != None and source_props['comp_vegamags']:
					ax.scatter(source_props['comp_seps'], source_props['comp_vegamags'], c='w', edgecolors='k', linewidths=2, s=50)
					for j, name in enumerate(source_props['comp_names']):
						ax.annotate(name, (source_props['comp_seps'][j], source_props['comp_vegamags'][j]), xytext=(5, 5), textcoords='offset points')
				ax.legend(frameon=False, fontsize=14)
				ax.set_ylim(ax.get_ylim()[::-1])
				ax.set_ylabel("Apparent Magnitude", fontsize=16)
				ax.set_xlabel('Separation (")', fontsize=16)
				ax.tick_params(which='both', direction='in', labelsize=14, axis='both', top=True, right=True)
				ax.set_title('{} // {}+{}'.format(target, filt.upper(), mask.upper()), fontsize=18)
				plt.savefig(plot_save_file, bbox_inches='tight', dpi=300)

	return contrast_curve_dict

