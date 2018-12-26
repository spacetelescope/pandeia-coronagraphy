"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
from sys import version_info

here = path.abspath(path.dirname(__file__))
NAME = 'jwst_pancake'

python_major = version_info[0]
if python_major >= 3:
    required=['numpy>=1.15','matplotlib>=2.2','pandeia.engine>=1.2', 'webbpsf>0.7', 
              'scikit-image>=0.14', 'pysynphot>=0.9', 'astropy>=2', 'photutils>=0.5', 
              'cython>=0.29', 'scipy>=1', 'poppy>0.7'],
else:
    required=['numpy>=1.15','matplotlib>=2.2','pandeia.engine>=1.2', 'webbpsf<0.7', 
              'scikit-image>=0.14', 'pysynphot>=0.9', 'astropy<3', 'photutils>=0.4', 
              'functools32>=3', 'cython>=0.29', 'scipy>=1', 'poppy<0.7'],

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Get version
with open(path.join(here, NAME, 'VERSION'), encoding='utf-8') as f:
    version = f.read().strip()

setup(
    name=NAME,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,

    description='A simple wrapper around the Pandeia engine to facilitate coronagraphy calculations for JWST',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/spacetelescope/pandeia_coronagraphy',

    # Author details
    author='Brian York',
    author_email='york@stsci.edu',

    # Choose your license
    license='BSD',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific :: Astronomy',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',

    ],

    # What does your project relate to?
    keywords='pandeia astronomy coronagraphy JWST',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),
    package_data={'jwst_pancake.templates' : ['*.json']},

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    #
    # Requirements info as of 2018-10-26:
    #   Actual things that we require:
    #       - numpy
    #       - matplotlib
    #       - pandeia
    #       - webbpsf
    #       - poppy
    #   Pandeia requires (but doesn't mention that it requires)
    #       - pyfftw
    #       - editing to work under python 3
    #       - photutils
    #   Errors of unknown sorts when compiling pyfftw require
    #       - cython
    #       - update of setuptools
    # So, the current actual way of installing correctly is:
    #   - create a new conda environment with python=3 and fftw (need conda forge for this)
    #   - make sure that numpy has been installed
    #   - make sure that cython is installed
    #   - make sure that setuptools are updated
    #   - install pip pyfftw via pip
    #   - install pip notebook in order to run notebooks
    #   - notebooks still can't import pandeia because apparently it can't see pyfftw in notebooks only.
    install_requires = required,
)
