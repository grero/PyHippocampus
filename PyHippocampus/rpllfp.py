import numpy as np 
import h5py as h5 
from scipy import signal 
from rplsplit import rplsplit

def rpllfp(auto = True, lowPassFrequency = [1, 150], LFPOrder = 8, resampleRate = 1000):
	# do this if rplraw.hdf5 already exists. 
	data = h5.File(file, 'w')
	analogData = np.array(data['analogData'])
	lanalogInfo = data['analogInfo']
	# Apply lowpass filter, create the filter and then see if it works. 
	sos = signal.butter(LFPOrder, lowPassFrequency, 'bandpass', fs = resampleRate, output = "sos")
	ldata = signal.sosfilt(sos, analogData)
	lanalogData = ldata 
	lanalogInfo['SampleRate'] = resampleRate
	lanalogInfo['MinVal'] = min(lanalogData)
	lanalogInfo['MaxVal'] = max(lanalogData)
	lanalogInfo['HighFreqCorner'] = lowPassFrequency[0] * resampleRate 
	lanalogInfo['LowFreqCorner'] = lowPassFrequency[1] * resampleRate 
	lanalogInfo['NumberSamples'] = len(lanalogData)
	lanalogInfo['HighFreqOrder'] = LFPOrder
	lanalogInfo['LowFreqOrder'] = LFPOrder
	lanalogInfo['ProbeInfo'].replace('raw', 'lfp') # replace the raw with lfp in the name 
	lanalogInfo['numSets'] = 1
	f = h5.File('rpllfp.hdf5', 'w')
	lanalogInfo = create_dataset('analogInfo', data = lanalogInfo)
	lanalogData = create_dataset('analogData', data = ldata)
	f.close()
	return "Low pass complete\n"

