Python code for analyzing hippocampus data

To install, launch Anaconda Navigator. On the left sidebar, select Environments. Select “base (root)” or another environment that you are using. Click on the triangle icon, and select “Open Terminal”. In the Terminal window, change to the PyHippocampus directory and do:

cd ~/Documents/Python/PyHippocampus
pip install -r requirements.txt
pip install -e .

Clone pyedfread for reading Eyelink files from GitHub to your computer by selecting Clone->Open in Desktop: https://github.com/nwilming/pyedfread

While still in the Terminal window, change directory to where the pyedfread code is saved, and do:

cd ~/Documents/Python/pyedfread
pip install .

Close the Terminal window, select Home in the sidebar of the Anaconda Navigator window, and launch Spyder. Type the following from the python prompt: 

import PyHippocampus as pyh

You should be able to use the functions by doing: 

rw = pyh.rplraw(...)
pyh.pyhcheck('hello')
uy = pyh.unity(...)
el = pyh.eyelink(...)

Test to make sure you are able to read EDF files: 
Change to a directory that contains EDF files, e.g.:

cd /Volumes/Hippocampus/Data/picasso-misc/20181105

Enter the following command: 

samples, events, messages = edf.pread('181105.edf', filter='all')
