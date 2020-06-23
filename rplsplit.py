import neo  
from neo.io import BlackrockIO 
import numpy as np 
import os 
import glob 
import h5py as h5 
from rplraw import rplraw 

''' 
The function must be run from within the date folder that contains the .nsx files. Splits the .ns5 file data into the individual channels. Also allows for the generation of the analogSignalTimes required to run rplparallel. 

This can allow the generation of .hdf5 file for any number of channels. This can be modified by the channels argument which is set to default as 'all', if a list is used instead, this would represent a selected number of channels. 
'''

def rplsplit(auto = True, channels = 'all', generateAnalogSignalTimes = False):
	'''Splits .ns5 file into individual channels, and then calls rplraw to save the appropriate data fields in the appropriate directory.'''
	ns5_file = glob.glob(".ns5")
	if len(ns5_file) > 1: 
		print("Too many .ns5 files. Do not know which one to use\n")
		return 
	reader = BlackrockIO(ns5_file[0])
	bl = reader.get_block()
	segment = bl.segments[0]
	chx = bl.channel_indexes[2] # For the raw data. 
	analogSignalList = [sig[:, chx.index] for sig in chx.analogsignals]
	analogSignalTimes = float(segment.analogsignals[2])
	analogSignalTimesFile = h5.File('analogSignalTimes.hdf5', 'w')
	analogSignalTimes = analogSignalTimesFile.create_dataset('analogSignalTimes', data = analogSignalTimes)
	analogSignalTimesFile.close()
	if generateAnalogSignalTimes: 
		return "Generated analogSignalTimes.hdf5\n"
	samplingRate = float(segment.analogsignals[2].sampling_rate)
	annotations = chx.annotations # For AnalogInfo, fields include, max, min, units, sampling rate, nev_high_frequency_type, nev_high_freq_order, nev_high_freq_corner, nev_low_frequency_type, nev_low_freq_order, nev_low_freq_corner
	probenames = chx.channel_names 
	def generateAnalogInfo(annotations, analogSignals, i):
		analogInfo = {}
		analogInfo['Units'] = 'uV'
		analogInfo['HighFreqCorner'] = annotations['nev_hi_freq_corner'][i]
		analogInfo['HighFreqOrder'] = annotations['nev_hi_freq_order'][i]
		analogInfo['HighFilterType'] = annotations['nev_hi_freq_type'][i]
		analogInfo['LowFreqCorner'] = annotations['nev_lo_freq_corner'][i]
		analogInfo['LowFreqOrder'] = annotations['nev_lo_freq_order'][i]
		analogInfo['LowFilterType'] = annotations['nev_lo_freq_type'][i]
		analogInfo['MaxVal'] = max(analogSignal)
		analogInfo['MinVal'] = min(analogSignal)
		analogInfo['NumberSamples'] = len(analogSignal)
		return analogInfo
	if channels == 'all': 
		for i in range(len(chx.index)): # i represents the index in the list here. 
			analogSignal = analogSignalList[0][i]
			arrayNumber = annotations['connector_ID'][i] + 1 
			analogInfo = generateAnalogInfo(annotations, analogSignal, i, arrayNumber)
			analogInfo['SampleRate'] = samplingRate
			analogInfo['ProbeInfo'] = probenames[i]
			rplraw(analogSignal, analogInfo, i)
	else: 
		for i in channels: # i represents the channel number here. 
			index = np.where(i == chx.index)[0][0]
			analogSignal = analogSignalList[0][index]
			arrayNumber = annotations['connector_ID'][index] + 1 
			analogInfo = generateAnalogInfo(annotations, analogSignal, index, arrayNumber)
			analogInfo['SampleRate'] = samplingRate
			analogInfo['ProbeInfo'] = probenames[index]
			rplraw(analogSignal, analogInfo, index)
	return "rplsplit complete\n"
