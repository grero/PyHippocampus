import numpy as np 
import h5py as h5 
import glob 
from rplsplit import rplsplit

def rplhighpass(auto = True, highPassFrequency = [500, 7500], highPassOrder = 8, resampleRate = 1000):
	# do this if rplraw.hdf5 already exists. 
	data = h5.File(file, 'w')
	analogData = np.array(data['analogData'])
	hanalogInfo = data['analogInfo']
	# Apply highpass filter 
	sos = signal.butter(highPassOrder, highPassFrequency, 'bandpass', fs = resampleRate, output = "sos")
	hdata = signal.sosfilt(sos, analogData)
	hanalogData = hdata 
	hanalogInfo['SampleRate'] = resampleRate
	hanalogInfo['MinVal'] = min(hanalogData)
	hanalogInfo['MaxVal'] = max(hanalogData)
	hanalogInfo['HighFreqCorner'] = highPassFrequency[0] * resampleRate 
	hanalogInfo['LowFreqCorner'] = highPassFrequency[1] * resampleRate 
	hanalogInfo['NumberSamples'] = len(hanalogData)
	hanalogInfo['HighFreqOrder'] = highPassOrder
	hanalogInfo['LowFreqOrder'] = highPassOrder
	hanalogInfo['ProbeInfo'].replace('raw', 'hp')  # replace the raw with hp in the name 
	hanalogInfo['numSets'] = 1
	f = h5.File('rplhighpass.hdf5', 'w')
	hanalogInfo = create_dataset('analogInfo', data = hanalogInfo)
	hanalogData = create_dataset('analogData', data = hdata)
	f.close()
	return "Highpass complete\n" 