Pandeia-Coronagraphy
=====

.. image:: screenshot.png
   :align: center
   :alt: Simulated NIRCam 210R scene with KLIP reference-subtraction 

Installation
----

It is highly recommended that you begin by installing `AstroConda <http://astroconda.readthedocs.io/en/latest/installation.html#install-astroconda>`_ (with Python 2.7) and then follow these the installation `instructions <https://gist.github.com/nmearl/c2e0a06d2d5a3715baf7d9486780dc08>`_ to install the Pandeia engine and the required reference files. Once Pandeia is set up, the following command will install this package:

 ``pip install git+git://github.com/kvangorkom/pandeia-coronagraphy.git``

_____

Alternatively, follow these step-by-step instructions:

1. If you don't already have Anaconda or Miniconda installed, download and install the Python 2.7 version `here <https://conda.io/miniconda.html>`_.

2. Add the AstroConda channel to your Conda channels: 

 ``conda config --add channels http://ssb.stsci.edu/astroconda``

3. Create a conda environment with the STScI software stack:

 ``conda create -n astroconda stsci python=2.7 numpy=1.11``

4. Activate this environment with ``source activate astroconda``. (NB: Conda is only compatible with a BASH shell.)

5. Install the Pandeia engine with this command: ``pip install pandeia.engine``. (You *should* already have the Pysynphot installed package at this point as well. If you don't, install it with ``pip install pysypnphot``. You can find your installed packages with ``conda list``.)

6. Download and unzip the `Pandeia data files <http://ssb.stsci.edu/pandeia/engine/1.0/pandeia_data-1.0.tar.gz>`_ and the `PySynphot data files <ftp://archive.stsci.edu/pub/hst/pysynphot/>`_. The entire PySynphot data file collection is quite large; you may be able to get away with only downloading the Pysynphot `Phoenix Models <ftp://archive.stsci.edu/pub/hst/pysynphot/synphot5.tar.gz>`_.

7. Add the following lines to your ~/.bashrc file (and ``source`` it after modifying):

 .. code-block:: bash

	export pandeia_refdata=/path/to/pandeia/data/directory
	export PYSYN_CDBS=/path/to/cdbs/directory
 
8. Finally, install the pandeia-coronagraphy package:

 ``pip install git+git://github.com/kvangorkom/pandeia-coronagraphy.git``

9. (Optional) Install `WebbPSF <https://pythonhosted.org/webbpsf/index.html>`_ with ``conda install webbpsf``. This is required only if you are interested in using higher-fidelity PSFs in your calculations; otherwise, the Pandeia engine relies on interpolations of a bundled library of precomputed PSFs. This functionality is documented `here <https://github.com/kvangorkom/pandeia-coronagraphy/blob/master/notebooks/nircam_on_the_fly_PSFs.ipynb>`_.

Getting Started
----

Once installation is complete, take a look at the provided `Jupyter notebooks <https://github.com/kvangorkom/pandeia-coronagraphy/tree/master/notebooks>`_ for examples of constructing a scene, setting instrument properties, running the Pandeia engine, and performing some basic post-processing. You can find a more complete description of the engine inputs `here <https://gist.github.com/nmearl/2465fe054a71ddaadba349398fa3e146#file-engine_input-md>`_ and outputs `here <https://gist.github.com/nmearl/2465fe054a71ddaadba349398fa3e146#file-engine_output-md>`_.
