Python code for analyzing hippocampus data

To install, launch Anaconda Navigator. On the left sidebar, select Environments. Select “base (root)” or another environment that you are using. Click on the triangle icon, and select “Open Terminal”. In the Terminal window, change to the PyHippocampus directory and do:

cd ~/Documents/Python/PyHippocampus

pip install -r requirements.txt

pip install -e .

Clone pyedfread for reading Eyelink files from GitHub to your computer by selecting Clone->Open in Desktop: 

https://github.com/nwilming/pyedfread

While still in the Terminal window, change directory to where the pyedfread code is saved, and do:

cd ~/Documents/Python/pyedfread
pip install .

You should also clone the following two repositories:

https://github.com/grero/DataProcessingTools

https://github.com/grero/PanGUI

Change to the directory where the code is saved, and install them using:

pip install -e .

Close the Terminal window, select Home in the sidebar of the Anaconda Navigator window, and launch Spyder. Type the following from the python prompt: 

import PyHippocampus as pyh

You should be able to use the functions by doing: 

pyh.pyhcheck('hello')

cd ~/Documents/Python/PyHippocampus

\# count number of items in the directory

df1 = pyh.DirFiles()

cd PyHippocampus

\# count number of items in the directory

df2 = pyh.DirFiles()

\# add both objects together

df1.append(df2)

\# plot the number of items in the first directory

df1.plot(i=0)

\# plot the number of items in the second directory

df1.plot(i=1)

Test to make sure you are able to read EDF files: 
Change to a directory that contains EDF files, e.g.:

cd /Volumes/Hippocampus/Data/picasso-misc/20181105

Enter the following command: 

samples, events, messages = edf.pread('181105.edf', filter='all')

You can create objects by doing:

rl = pyh.RPLParallel()

uy = pyh.Unity()

el = pyh.Eyelink()

You can create plots by doing:

rp = PanGUI.create_window(rl)

up = PanGUI.create_window(uy)

ep = PanGUI.create_window(el)
