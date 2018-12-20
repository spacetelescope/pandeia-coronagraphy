from __future__ import absolute_import

from . import scene
from . import analysis
from . import engine
from . import transformations
from . import utilities

import os, sys
__version__ = "UNDEFINED"
try:
    version_file = os.path.join(os.path.split(__file__)[0], "VERSION")
    with open(version_file, "r") as inf:
        __version__ = inf.readline().strip()
except Exception as e:
    sys.err.write("Unable to find pancake version file!\n")

import os
tmp = os.getenv('pandeia_refdata')
if tmp is None:
    raise RuntimeError("ERROR - you need to set the environment variable pandeia_refdata or calculations will not work")
del tmp

