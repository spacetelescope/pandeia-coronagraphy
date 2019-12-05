============
Introduction
============

The JWST Pandeia Coronagraphy Advanced Kit and ETC.
----------------------------------------------------

*Authors*: Bryony Nickson (STScI)

*Contributors*:

**!! Under Development !!**

The PanCAKE toolkit contains a set of python-based tools for planning coronagraphic observations with
JWST.

Why PanCAKE?
'''''''''''''

There are several limitations to Pandeia/ the JWST web-based ETC that can have an important impact when planning JWST coronagraphic observations:

 - The ETC supposes a perfect centering (target acquisition) of all stars, and therefore calculations may be too optimistic.
 - The ETC does not account for spectral mismatch (only photometrically) between target and PSF reference stars, which can have an important impact on NIRCam coronagraphic observations.
 - Pandeia uses a pre-computed PSF library from WebbPSF with a discrete number of angular separations. This sparse spatial sampling can result in inaccurate calculations in the speckle limited regime close to the coronagraph centers (typically at seperations < 1 arcseconds).
 - The ETC does not support small grid dithers.

PanCake, however, allows users to:

  - Circumvent the use of Pandeia's precomputed PSF library by generating PSFs on the fly in WebbPSF.
  - Capture the effect of target acquisition error.
  - Customize the wavelength sampling.
  - Manually perform post-processing (subtracting registered and scaled reference PSFs from the target images).
  - Generate contrast curves.
  - Implement custom small grid dithers and positioning.

What PanCAKR CANNOT do:

  - Simulate the entire instrument fields of view. (Future work)
  - Simulate ring-like features or disks (Future work)
  - Include optical field distortion, intrapixel response variations, other detector systematic noise, or the effects of spacecraft jitter or drift.


.. admonition::

    Note that PanCAKE enables more modes than are officially allowed by the Observatory, (ie., filter + coronagraphic combinations, subarray sizes, etc.). Functionality that can be performed with PanCAKE may not necessarily be supported by the instrument. Refer to the NIRCam and MIRI observing modes' `documentation <https://jwst-docs.stsci.edu/>`_.

Similar to some of its dependencies, PanCAKE requires a host of input data files in order to generate simulations. Due to the size of these files, they are not included with this source distribution. Please see the documentation for instructions on how to to download the required data files.
