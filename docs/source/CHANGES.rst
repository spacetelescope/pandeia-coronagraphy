#################
2.0 (unreleased)
#################

--------------------

New Features 
*************

    * Change of package name to 'jwst_pancake' [1ba7a12]

** jwst_pancake.utilities ** (New!) :
    * New function, containing a number of PanCAKE utilitize which may be useful in running PanCAKE.
        * Added to __init__.py [2b1693a]

** jwst_pancake.setup ** :
    * Include installation requirememnts for both Python 2 and Python 3 are included.  

** jwst_pancake.scene ** : can now apply target acquisition errors to each point in SGDs. 

**jwst_pancake.pandeia_subclasses** : 
    * Added pupil throughput item for Pandeia 1.3.
    * Added a new attribute to ``CoronagraphyDetectorSignal`` for Pandeia 1.3.

** jwst_pancake.engine ** : 
    * Added convinience functions:
        * target-only calculation of both target and reference scene.


API Changes
*************

    * Includes new notebooks, compatable with Pandeia 1.3 [919db07], including:
        * A new installation notebook. 
        * MIRI Pandeia/ Pancake comparision notebook. 
        * NIRCam Pandeia/ Pancake comparision notebook.
    
    * Updates to MIRI notebooks. [f915bfc]
    * Updates to NIRCam notebooks. [f915bfc]

    **jwst_pancake.pandeia_subclasses** : 
        * Removed extraneous print function.


Bug Fixes 
*************

** jwst_pancake.setup ** :
    * Fixed required installation. [3802834]
    * Updated photoutils version requirement to version 0.4.
    * Inclusion of poppy package with setup to avoid potential versioning mismatch.

** jwst_pancake.engine ** : Updated to work with Pandeia 1.3 and WebbPSF 0.8.

** jwst_pancake.scene ** : Updated to work with Pandeia 1.3 and Webb PSF 0.8.


Other Changes and Additions
****************************

