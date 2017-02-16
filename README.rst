Pandeia-Coronagraphy
=====

.. image:: screenshot.png
   :align: center
   :alt: Simulated NIRCam 210R scene with KLIP reference-subtraction 

Installation
----

It is highly recommended that you begin by installing `AstroConda <http://astroconda.readthedocs.io/en/latest/installation.html#install-astroconda>`_ (with Python 2.7) and then follow these `instructions <https://gist.github.com/nmearl/c2e0a06d2d5a3715baf7d9486780dc08>`_ (just the "Installation Procedure" section) to install the Pandeia engine and the required reference files.

Once Pandeia is set up, the following command will install this package:

``pip install git+git://github.com/kvangorkom/pandeia-coronagraphy.git``

Getting Started
----

Take a look at the provided `Jupyter notebooks <https://github.com/kvangorkom/pandeia-coronagraphy/tree/master/notebooks>`_ for examples of constructing a scene, setting instrument properties, running the Pandeia engine, and performing some basic post-processing. You can find a more complete description of the engine inputs `here <https://gist.github.com/nmearl/2465fe054a71ddaadba349398fa3e146#file-engine_input-md>`_ and outputs `here <https://gist.github.com/nmearl/2465fe054a71ddaadba349398fa3e146#file-engine_output-md>`_.
