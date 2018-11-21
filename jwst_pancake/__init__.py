from __future__ import absolute_import

from . import scene
from . import analysis
from . import engine
from . import transformations
from . import utilities


import os
tmp = os.getenv('pandeia_refdata')
if tmp is None:
    raise RuntimeError("ERROR - you need to set the environment variable pandeia_refdata or calculations will not work")
del tmp

