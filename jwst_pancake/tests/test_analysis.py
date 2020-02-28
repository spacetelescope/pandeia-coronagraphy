import numpy as np
import matplotlib.pyplot as plt
import copy

import jwst_pancake.analysis
import poppy
import webbpsf

def test_register_to_target(plot=True, box=50):
    """ Test the register_to_target function, using a simple Airy function.

	See also test_register_images_nircam for a harder test case with NIRCam images

    Using Airy functions, this randomly offset the 'reference' by a moderately large
	number of integer pixels, and tests we can realign after that.
    """
    size = 512
    center = (size-1)/2
    dx = np.random.randint(-box, box)
    dy = np.random.randint(-box, box)
    target = poppy.misc.airy_2d(center=(center, center))   # centered PSF we will shift to match
    ref = poppy.misc.airy_2d(center=(center+dx, center+dy))  # randomly offset PSF


    aligned_ref, offy, offx, scale = jwst_pancake.analysis.register_to_target(ref, target, return_fit=True,
                                                                                     rescale_reference=False)

    box_min = int(size/2-box)
    box_max = int(size/2+box)

    target_zeromean = target - np.nanmean(target)  # The aligned reference is set to have zero mean,
                                                  # so we must do the same to the target for a fair comparison.

    difference = target_zeromean - aligned_ref

    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(141)
        plt.imshow(target)
        plt.title("Centered Target\nat ({}, {})".format(center, center))
        plt.subplot(142)
        plt.imshow(ref)
        plt.title("Offset Reference\nat ({:.2f}, {:.2f})".format(center+dx, center+dy))
        plt.subplot(143)
        plt.imshow(aligned_ref)
        plt.title("Reference Aligned\nShifted by ({:.2f}, {:.2f})".format(offx, offy))
        plt.subplot(144)
        plt.imshow(target-aligned_ref)
        plt.title("Diff of \nTarget - Aligned")

        plt.plot( [box_min, box_min, box_max, box_max, box_min],
                  [box_min, box_max, box_max, box_min, box_min], ls=':', color='white')
        plt.figure()
        plt.imshow(difference[box_min:box_max, box_min:box_max], vmin=-1e-6, vmax=1e-6)
        plt.colorbar()
        plt.title("Difference inside box\non different color scale!")



    print("Sum of diff image:", np.abs(difference[box_min:box_max, box_min:box_max]).sum())
    print("Box coords:", box_min, box_max)
    print("Diff mean inside box:", difference[box_min:box_max, box_min:box_max].mean())
    assert np.abs(difference[box_min:box_max, box_min:box_max]).sum() < target.sum()/1e4, "Image difference too large"
    return difference

def test_register_images_nircam(offset_x=None, offset_y=None, fov=6, plot=False):
    """ Test the image registration works properly, using simulated NIRCam images
	This is a more complicated and more realistic test case than in the
	test_register_to_target function.

	"""

    # by default test for up to 1 pixel offset in any direction
    if offset_x is None:
        offset_x = np.random.uniform(low=-1, high=1)
    if offset_y is None:
        offset_y = np.random.uniform(low=-1, high=1)

    nc = webbpsf.NIRCam()

    nc.filter='F335M'
    nc.image_mask='MASK335R'
    nc.pupil_mask='MASKRND'

    print("computing PSFs. Reference PSF offset by ({:.2f}, {:.2f}) pix = ({:.4f}, {:.4f}) arcsec".format(
        offset_x, offset_y, offset_x*nc.pixelscale, offset_y*nc.pixelscale))
    target = nc.calc_psf(nlambda=1, fov_arcsec=fov )

    nc.options['source_offset_x']= offset_x * nc.pixelscale
    nc.options['source_offset_y']= offset_y * nc.pixelscale
    reference_offset = nc.calc_psf(nlambda=1, fov_arcsec=fov )


    reference_aligned = copy.deepcopy(reference_offset)
    target_zeromean = copy.deepcopy(target)

    print("Aligning...")
    for ext in [0,1]:
        reference_aligned[ext].data, offx, offy, scale = jwst_pancake.analysis.register_to_target(
                                                                       reference_offset[ext].data, # reference
                                                                       target[ext].data,  # target
                                                                       verbose=True,
                                                                       return_fit=True,
                                                                       rescale_reference=False)
        print("  Ext {} offset is {:.4f}, {:.4f} pixels".format(ext, offx, offy))

        if ext == 0:
            # Compare the measured offset to the oversampled desired offset
            sampling = reference_aligned[ext].header['OVERSAMP']
        else:
            sampling = 1
        diffx = offx /sampling - offset_x
        diffy = offy /sampling - offset_y
        print("  Measured offset vs. applied offset residual: ({:.3f}, {:.3f}) pix".format(diffx, diffy))
        assert np.abs(diffx) < 0.05, "Residual offset in X direction too large: {} pix".format(diffx)
        assert np.abs(diffy) < 0.05, "Residual offset in Y direction too large: {} pix".format(diffx)


        # Note, the aligned PSF is set to have zero mean, so do the same for the target
        target_zeromean[ext].data -= np.nanmean(target_zeromean[ext].data)
        #print("  Set target image's mean = 0 , too")

    totdiff_before = np.abs(target[1].data - reference_offset[1].data).sum()
    totdiff_after  = np.abs(target_zeromean[1].data - reference_aligned[1].data).sum()

    print("Total residual before alignment: ", totdiff_before)
    print("Total residual after alignment: ", totdiff_after)
    print("Improvement factor: ", totdiff_after/totdiff_before)

    assert totdiff_after < totdiff_before/4, "Image residuals didn't get enough better after alignment."

    if plot:
        # Plot the before
        plt.figure()
        webbpsf.display_psf_difference(target, reference_offset, title='Diff BEFORE realignment, Ext 1',
                                       normalize=False, vmax=1e-5,
                                       ext1=1, ext2=1)


        # plot the after
        plt.figure()
        webbpsf.display_psf_difference(target_zeromean, reference_aligned, title='Diff after realignment, Ext 1',
                                   normalize=False, vmax=1e-5,
                                   ext1=1, ext2=1)



