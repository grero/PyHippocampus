import neo  
from neo.io import BlackrockIO 
import numpy as np 
import os 
import glob 
import h5py as h5 

def rplsplit():
	'''Splits .ns5 file into individual channels, and then calls rpl_raw to save the appropriate data fields in the appropriate directory.'''
	ns5_file = glob.glob(".ns5")
	if len(ns5_file) > 1: 
		print("Too many .ns5 files")
		return 
	reader = BlackrockIO(ns5_file[0])
	bl = reader.get_block()
	segment = bl.segments[0]
	chx = bl.channel_indexes[2] # For the raw data. 
	analogSignalList = [sig[:, chx.index] for sig in chx.analogsignals]
	analogSignalTimes = float(segment.analogsignals[2])
	rplparallel(analogSignalTimes)
	samplingRate = float(segment.analogsignals[2].sampling_rate)
	annotations = chx.annotations # For AnalogInfo, fields include, max, min, units, sampling rate, nev_high_frequency_type, nev_high_freq_order, nev_high_freq_corner, nev_low_frequency_type, nev_low_freq_order, nev_low_freq_corner
	probenames = chx.channel_names 
	for i in range(len(chx.index)):
		analogSignal = analogSignalList[0][i]
		analogInfo = {}
		analogInfo['MaxVal'] = max(analogSignal)
		analogInfo['MinVal'] = min(analogSignal)
		analogInfo['SamplingRate'] = samplingRate
		analogInfo['Units'] = 'uV'
		analogInfo['HighFrequencyCorner'] = annotations['nev_high_freq_corner'][i]
		analogInfo['HighFrequencyOrder'] = annotations['nev_high_freq_order'][i]
		analogInfo['HighFilterType'] = annotations['nev_high_freq_type'][i]
		analogInfo['LowFrequencyCorner'] = annotations['nev_low_freq_corner'][i]
		analogInfo['LowFrequencyOrder'] = annotations['nev_low_freq_order'][i]
		analogInfo['LowFilterType'] = annotations['nev_low_freq_type'][i]
		analogInfo['ProbeInfo'] = probenames[i]
		analogInfo['NumberOfSamples'] = len(analogSignal)
		rplraw(analogSignal, analogInfo, i)
		return 

def rplraw(analogSignal, analogInfo, channelNumber):
	'''Generates a rplraw.hdf5 file in the corresponding channel directory which contains the following fields: (i) analogInfo and (ii) analogData.'''
	# data = {'analogData': analogSignal, 'analogInfo':analogInfo}
	filesInDirectory = os.listdir('.')
	if 'session01' not in filesInDirectory: 
		os.mkdir('session01')
	os.chdir('session01') 
	if 'channel{:02d}'.format(channelNumber) not in os.listdir('.'):
		os.mkdir('channel{:02d}'.format(channelNumber))
	os.chdir('channel{:02d}'.format(channelNumber))	
	f = h5.File('rplraw.hd5f', 'w')
	analogSignal = f.create_dataset('analogSignal', data = analogSignal)
	analogInfo = f.create_dataset('analogInfo', data = analogInfo)
	f.close()
	os.chdir('..')
	return 

def rplparallel(analogSignalTimes):
	'''Generates a rplparallel.hdf5 file in the session01 directory which contains the following fields: (i) markers, (ii) timestamps, (iii) sample rate, (iv) trialIndices and (v) session start seconds.'''
	nev_file = glob.glob("*.nev")
	if len(nev_file) > 1: 
		print("Too many .nev files")
		return 
	reader = BlackrockIO(nev_file[0])
	ev_rawtimes, _, ev_markers = reader.get_event_timestamps()
	ev_times = reader.rescale_event_timestamp(ev_rawtimes, dtype = "float64") 
	session_start_sec = ev_times[0]
	markers = ev_markers[::2][1:] # Remove the 84 and 0 markers 
	timeStamps = ev_times[::2][1:] # Remove the time corresponding to the 84 and 0 markers. 
	markers = np.array([np.array(markers[i:i+3]) for i in range(0, len(markers), 3)])
	timeStamps = np.array([np.array(timeStamps[i:i+3]) for i in range(0, len(timeStamps), 3)])
	samplingRate = float(analogSignalTimes.sampling_rate) 
	trialIndices = []
	for i in range(len(timeStamp)):
	    temp = []
	    for j in range(len(timeStamps[i])):
	        index = np.where(analogSignalTimes == timeStamps[i][j])[0][0]
	        temp.append(index)
	    trialIndices.append(np.array(temp))
	trialIndices = np.array(trialIndices)
	f = h5.File('rplparallel.hdf5', 'w')
	markers = f.create_dataset('markers', data = markers)
	session_start_sec = f.create_dataset('session_start_sec', data = session_start_sec)
	timeStamps = f.create_dataset('timeStamps', data = timeStamps)
	samplingRate = f.create_dataset('samplingRate', data = samplingRate)
	trialIndices = f.create_dataset('trialIndices', data = trialIndices)
	f.close() 
	return "RPLParallel .hdf5 File created"

def main():
	rplsplit()

if __name__ == "__main__":
	main()
