from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
        # ext_modules = cythonize("ddm_data_simulation1.pyx", annotate=True),
        # ext_modules = cythonize("make_data_wfpt1.pyx", annotate=True),
        # ext_modules = cythonize(["cddm_data_simulation.pyx", "cdweiner.pyx"], annotate=True),
        ext_modules = cythonize("cdwiener.pyx", annotate=True),
        # ext_modules = cythonize("wfpt.pyx", language="c++"),
        include_dirs = [numpy.get_include()]
    )
