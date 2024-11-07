from setuptools import setup
import os
import json

#get the version number
dd,ff = os.path.split(__file__)
dd = os.path.join(dd, "PyHippocampus")
_meta = json.load(open(os.path.join(dd, 'version.json')))
__version__ = _meta["version"]

setup(
	name="PyHippocampus",
	version=__version__,
	description="""Tools for processing NHP hippocampus data""",
	url="https://github.com/shihchengyen/PyHippocampus.git",
	author="Shih-Cheng Yen & friends",
	author_email="shihcheng@nus.edu.sg",
	packages=["PyHippocampus"],
	scripts=["PyHippocampus/si_sorting.py"],
)
