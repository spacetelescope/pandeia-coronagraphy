.. image:: jwst-pancake_logo_full.png
    :align: center
    :alt: jwst-pancake

---------------------------------------

.. _install:

##############################
Requirements & Installation
##############################

This page provides information on how to install panCAKE on your system, including all necessary additional python modules.

********************
Installing PanCAKE
********************

Recommended method: Installing via AstroConda
===============================================

For ease of installation, it is highly recommended using `AstroConda <https://astroconda.readthedocs.io/en/latest/>`_---an astronomy-optimized software distribution for scientific Python built on `Anaconda <https://www.anaconda.com/>`_. 
AstroConda requires the `Conda <https://docs.conda.io/en/latest/>`_ package and environment manager, which is included in all versions of Anaconda and `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. If you do not have Anaconda or Miniconda already installed, download either option from `the Conda download page <https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/download.html>`_ and follow the documentation for installation.

.. note::

    Conda requires `BASH <https://tiswww.case.edu/php/chet/bash/bashtop.html>`_, or a BASH-compatible shell in order to function correctly. If your default shell environment is not BASH, execute
    ``bash -l`` before proceeding.

In order to install packages directly from the AstroConda channel, you will need to add it to Conda’s channel search path.
From a terminal with access to the conda environment, type the following: 
    
.. code:: bash

    $ conda config --add channels http://sbb.stsci.edu/astroconda.
    # Writes changes to ~/.condarc
        
which will configure Conda to pull from the AstroConda repository. Create a new environment with the |stsci|_ (which contains nearly all of the software provided by STScI) installed.
For example, to create an environment named "pancake-env" working under python 3, type:

.. code:: bash

    $ conda create -n pancake-env stsci python=3 

Activate the PanCAKE conda environment by running the :py:mod:`$ source activate pancake-env` and install the Pandeia engine using:

.. sidebar:: Installing Pandeia

    Instructions for downloading Pandeia are also provided in the following `Installing Pandeia article <https://jwst-docs.stsci.edu/display/JPP/Installing+Pandeia>`_.

.. code:: bash

    (pancake-env)$ pip install pandeia.engine==1.4
    
You should already have the |Pysynphot|_ installed (through the STScI software stack), 
but if you do not, install it with :py:mod:`pip install pysynphot`;
You can generate a list of installed packages with :py:mod:`conda list`.

.. |stsci| replace:: :py:mod:`stsci` package
.. _stsci: https://astroconda.readthedocs.io/en/latest/package_manifest.html
.. |pysynphot| replace:: :py:mod:`pysynphot` package
.. _pysynphot: https://pysynphot.readthedocs.io/en/latest/

To install |WebbPSF|_ (along with all its dependencies and required reference data), type the following::

    (pancake_env)$ conda install webbpsf

Lastly, install the |jwst-pancake|_ package from GitHub::

    $ pip install git+git://github.com/spacetelescope/jwst-pancake.git


.. |WebbPSF| replace:: :py:mod:`webbpsf` package
.. _WebbPSF: https://webbpsf.readthedocs.io/en/latest/    

.. |jwst-pancake| replace:: :py:mod:`jwst-pancake` package
.. _jwst-pancake: https://github.com/spacetelescope/pandeia-coronagraphy/  

.. tip::

    If you wish to use `jupyter notebooks <https://jupyter.org/>`_ with PanCAKE, it would be useful to install the :py:mod:`nb_conda_kernels` conda package, 
    in order to ensure that you can choose which python installation your notebook is using. This can be done by typing :py:mod:`$ conda install nb_conda_kernels`.


------------------------------



Installing Without AstroConda
===============================

It is strongly recommended you install PanCAKE via AstroConda. Whilst some information is provided below for installing PanCAKE without AstroConda,
it is entirely unsupported, and any issues you encounter will likely be much more difficult to resolve. 

Installing with Anaconda
-------------------------

Begin by creating an Anaconda environment::

    $ conda create -n pancake-env fftw

.. note::

    If you do not have :py:mod:`fftw` installed, and do not wish to install it yourself, you should add 
    `conda-forge <https://conda-forge.org/>` to your available channels by entering the following::
    
        $ conda config --add channels conda-forge

    into the command line.        

Activate the new anaconda environment and clone and intall PanCAKE locally using pip::

    (pancake-env)$ git clone https://github.com/spacetelescope/jwst-pancake.git
    (pancake-env)$ pip install jwst-pancake/

This will automatically install (almost) all of the needed Python packages. However, Pandeia has an undeclared dependency on :py:mod:`pyfftw`, 
so you will be required to type::

    (pancake-env)$ pip install pyfftw

at the command line. Ensure that your :ref:`environment variables <data_install>` are set up, and then you should be able to run panCAKE.


Installing with pip
--------------------

PanCAKE and its dependecies may also be installed using `pip <https://pypi.org/project/pip/>`_ the `package installer <https://packaging.python.org/guides/tool-recommendations/>`_
for Python. Pip supports installing from PyPI as well as cloning over `GitHub <https://github.com/>`_. 

Pandeia, WebbPSF and their dependencies can be installed from the `Python Package Index <https://pypi.org/>`_ in the usual manner for
Python packages::
  
    pip install pandeia.engine==1.4
    pip install pysynphot
    pip install WebbPSF

and PanCAKE can be installed with the following command::

    pip install git+git://github.com/spacetelescope/jwst-pancake.git


--------------------------


.. _data_install:

***********************************
Installing the Required Data Files
***********************************

PanCAKE relies on a number of other packages which require external data files to run. These include Pandeia (which relies on its own data files and the CBDS data) and WebbPSF.

Pandeia relies on two sets of configration files. The first set of files is for Pandeia itself, and can be downloaded here:

+----------------------------------+---------------------------------------------------------------------------------------------+
| **Pandeia v1.4 reference data**: | `pandeia-refdata-v1p4 <https://stsci.app.box.com/v/pandeia-refdata-v1p4>`_ [approx 1.94 GB] |  
+----------------------------------+---------------------------------------------------------------------------------------------+

Download and unpack these files to an appropriate location; we recommend "``$HOME/data/pandeia``".

.. warning::

    Backwards compatibility with Pandeia data files earlier than v1.3 is deprecated and was removed in v1.4.

The second set of data files are for `pysynphot <https://pysynphot.readthedocs.io/en/latest/>`_; Pandeia uses pysynphot internally 
for creating reference spectra. The :py:mod:`pysynphot` 
reference files may be downloaded here:

+-------------------------------+--------------------------------------------------------------------------------------------------------+
| **pysynphot reference data**: | `archive.stsci.edu/pub/hst/pysynphot/ <http://archive.stsci.edu/pub/hst/pysynphot/>`_ [approx 1.66 GB] | 
+-------------------------------+--------------------------------------------------------------------------------------------------------+
Note that the `tar.gz` files will untar into a directory structure that looks like "*grp/hst/cdbs*", with the actual files in an assortment 
of directories under "cdbs". You will need to consolidate the multiple structures into a single directory structure under cdbs in order to allow pysynphot (and pandeia)
to properly detect the reference files. 

.. note:: 

    If you're on the STScI network, you can skip this download and point the :py:mod:`PYSYN_CDBS` environment 
    variable to the CDBS directory on central store instead: ":py:mod:`/grp/hst/cdbs`".


WebbPSF also relies on a set of data files, containing such information as the JWST pupil shape, instrument throughputs, and aperture positions. 
In order to run WebbPSF, you must download the data from 

+----------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| **WebbPSF v0.8 reference data**: | `webbpsf-data-0.8.0.tar.gz <http://www.stsci.edu/~mperrin/software/webbpsf/webbpsf-data-0.8.0.tar.gz>`_ [approx 240 MB] |
+----------------------------------+-------------------------------------------------------------------------------------------------------------------------+ 
and untar it into a directory of your choosing.

.. note:: 

    The required data files for WebbPSF may also be accessed from the Central Storage network. Point the :py:mod:`WEBBPSF_PATH` environment variable to the
    ":py:mod:`grp/jwst/ote/webbpsf-data`" directory.


Once the reference files are downloaded, you must tell Pandiea and WebbPSF to find them using the ``pandeia_refdata``,
``PYSYN_CDBS`` and ``WEBBPSF_PATH`` environment variables. Set each of the environment variables to point to the correct directories, e.g.:

.. code:: bash

    export pandeia_refdata=/path/to/pandeia/data/directory
    export PYSYN_CDBS=/path/to/data/directory/grp/hst/cdbs
    export WEBBPSF_PATH=/path/to/webbpsf-data
   

In order to set these environemt variables, you can do one of the following:

 1. In anaconda/astroconda, :ref:`edit your activate and *deactivate scripts* <conda_scripts>`.
 2. By setting your :py:mod:`.profile` file to include the necessary environment variables.
 3. By :ref:`setting the environment variables in your script using the os module <os_module>`.

.. _Transition_

.. _conda_scripts: 

Conda Activation scripts    
==========================

Anaconda/ Astroconda allows for a shell script to run when a particular environment is activated or deactivated. 
The activation script should be located at :py:mod:`/path/to/anaconda/envs/your_env/etc/conda/activate.d/env_vars.sh`,
whilst the deactivation script should be located at :py:mod:`path/to/anaconda/envs/your_env/conda/deactivate.d/env_vars.sh`.
In both cases, :py:mod:`/path/to/anaconda/` is the path to your anaconda installation, and :py:mod:`your_env` is the name of the envirnment you 
are using to run PanCAKE.

**Activation:** the activation script sets the values of environment variables. 
For instance, for a pancake installation where you have a directory named "data" in your home directory, 
and that directory contains the CDBS data tree (named cdbs), the webbpsf data tree (named webbpsf-data) 
and the pandeia data tree (named pandeia_data), the activation env_vars.sh script would be:

.. code:: bash 

    !/bin/sh

    export PYSYN_CDBS=$HOME/data/cdbs
    export WEBBPSF_PATH=$HOME/data/webbpsf-data
    export pandeia_refdata=$HOME/data/pandeia_data


**Deactivation**: the deactivation script should unset all of the environment variables set by the activation script. 
So, for the above activation script example, the deactivation `env_vars.sh` script would be:

.. code:: bash

    #!/bin/sh
    
    unset PYSYN_CDBS
    unset WEBBPSF_PATH
    unset pandeia_refdata    


.. _os_module:

OS Module Environment
=======================

The python |os|_ can be used to set environment variables before PanCAKE (or one of its dependencies) is imported. 
The following code shows an example:

.. |os| replace:: :py:mod:`os` module
.. _os: https://docs.python.org/3/library/os.html

.. code::

    import os
    
    os.environ['PYSYN_CDBS'] = '$HOME/data/cdbs'
    os.environ['WEBBPSF_PATH'] = '$HOME/data/webbpsf-data'
    os.environ['pandeia_refdata'] = '$HOME/data/pandeia_data'

In the above example, you are assumed to have a directory named data in your home directory, 
which contains the CDBS data tree (named ``cdbs``), the webbpsf data tree (named ``webbpsf-data``), 
and the pandeia data tree (named ``pandeia_data``).    

----------------------

************************
Software Requirements
************************

Known Compatible Versions
==========================

PanCAKE has been tested sucessfully with the following packages:



Required Python Version
--------------------------

PanCAKE may be installed with either Python 2 or Python 3; however WebbPSF 0.8 and higher require Python 3.5 or higher.

 - Python 3, Pandeia 1.4, Webbpsf 0.8, Astropy 3
 - Python 3, Pandeia 1.3, Webbpsf 0.8, Astropy 3
 - Python 3, Pandeia 1.3, Webbpsf 0.6, Astropy 3
 - Python 3, Pandeia 1.3, Webbpsf 0.8, Astropy 3
 - Python 3, Pandeia 1.2, Webbpsf 0.6, Astropy 2
 - Python 2, Pandeia 1.2, Webbpsf 0.6, Astropy 2

**Required Python Packages**: 

 - :py:mod:`NumPy`
 - :py:mod:`SciPy` 
 - :py:mod:`matplotlib`
 - :py:mod:`Pandeia.engine`

**Recommended Python Packages**:

 - :py:mod:`pysynphot`, enable the simulation of PSFs with proper spectral response to realistic source spectra. Without this PSF fidelity is reqduced.
 - :py:mod:`WebbPSF`
 - :py:mod:`POPPY`

**Optional Python packages:**:

Some calculations with WebbPSF can benefit with the optional package :py:mod:`psutil`, but this is not needed in general. 


Contributing to PanCAKE
------------
The PanCAKE source code repository is hosted at GitHub. Users may clone or fork in the usual manner. Pull requests with enhancements are welcomed. 