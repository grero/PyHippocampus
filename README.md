Python code for analyzing hippocampus data

## Installation

### Create a new conda environment

From terminal do

```bash
conda create -n hippocampus python==3.9
conda activate hipoocampus
```

Then, clone this repository, for instance in `~/Documents/python/PyHippocampus`

```bash
cd ~/Documents/python
git clone https://github.com/grero/PyHippocampus.git PyHippocampus
```

Install the requirements

```bash
pip install -r requirements.txt
```

and install the package itself into your python environment

```bash
pip install -e .
```

Clone pyedfread for reading Eyelink files from GitHub to your computer by selecting Clone->Open in Desktop: 

https://github.com/nwilming/pyedfread

or using terminal,


```bash
pip install git+https://github.com/nwilming/pyedfread
```

You should also clone the following two repositories:

```bash
pip install git+https://github.com/grero/DataProcessingTools
pip install git+https://github.com/grero/PanGUI
```

Close the Terminal window, select Home in the sidebar of the Anaconda Navigator window, and launch Spyder. Type the following from the python prompt: 

```python
import PyHippocampus as pyh
```

You should be able to use the functions by doing: 

```python
pyh.pyhcheck('hello')

cd ~/Documents/Python/PyHippocampus

# count number of items in the directory

df1 = pyh.DirFiles()

cd PyHippocampus

# count number of items in the directory

df2 = pyh.DirFiles()

# add both objects together

df1.append(df2)

# plot the number of items in the first directory

df1.plot(i=0)

# plot the number of items in the second directory

df1.plot(i=1)
```

Test to make sure you are able to read EDF files: 
Change to a directory that contains EDF files, e.g.:

```python
cd /Volumes/Hippocampus/Data/picasso-misc/20181105
```

Enter the following command: 

```python
samples, events, messages = edf.pread('181105.edf', filter='all')
```

You can create objects by doing:

```python
rl = pyh.RPLParallel()

uy = pyh.Unity()

el = pyh.Eyelink()
```

You can create plots by doing:

```python
rp = PanGUI.create_window(rl)

up = PanGUI.create_window(uy)

ep = PanGUI.create_window(el)
```
