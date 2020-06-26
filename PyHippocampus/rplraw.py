import neo  
from neo.io import BlackrockIO 
import numpy as np 
import os 
import h5py as h5 

''' 
To be run from within the date folder that contains the raw .nsx files. Four fields to be specified which allow for the appropriate folders to be created or manouvered into to create the appropriate rplraw.hdf file for the channel. 
'''

def rplraw(analogSignal, analogInfo, channelNumber, arrayNumber):
	'''Generates a rplraw.hdf5 file in the corresponding channel directory which contains the following fields: (i) analogInfo and (ii) analogData.'''
	# data = {'analogData': analogSignal, 'analogInfo':analogInfo}
	if 'session01' not in os.listdir('.'): # Beginning from the date directory check the presence of the session directory 
		os.mkdir('session01')
	os.chdir('session01') 
	if 'array{:02d}'.format(arrayNumber) not in os.listdit('.'): # check whether the relevant array directory exsits 
		os.mkdir('array{:02d}'.format(arrayNumber))
	os.chdir('array{:02d}'.format(arrayNumber))
	if 'channel{:02d}'.format(channelNumber) not in os.listdir('.'): # check whether the relevant channel directory exists 
		os.mkdir('channel{:02d}'.format(channelNumber))
	os.chdir('channel{:02d}'.format(channelNumber))	
	# Write the file to the appropriate channel directory 
	f = h5.File('rplraw.hd5f', 'w') 
	analogSignal = f.create_dataset('analogSignal', data = analogSignal)
	analogInfo = f.create_dataset('analogInfo', data = analogInfo)
	f.close()
	return "rplraw.hdf5 for channel number {} created".format(channelNumber)