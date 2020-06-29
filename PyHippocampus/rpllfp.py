import numpy as np 
import h5py as h5 
from filters import lowPassFilter 

def rpllfp(lowPassFrequency = [1, 150], LFPOrder = 8, resampleRate = 1000, display = False, saveFig = False):
	file = 'rplraw.hdf5'
	data = h5.File(file, 'r')
	analogData = np.array(data['analogSignal'])
	samplingRate = np.array(data['SampleRate'])
	lfpData, resampleRate = lowPassFilter(analogData, samplingRate = samplingRate, resampleRate = resampleRate, LFPOrder = LFPOrder, lowFreq = lowPassFrequency[0], highFreq = lowPassFrequency[1], display = False, saveFig = False)
	lfp_file = h5.File('rpllfp.hdf5', 'w')
	lanalogData = lfp_file.create_dataset('analogData', data = lfpData)
	sampleRate = lfp_file.create_dataset('SampleRate', data = resampleRate)
	minVal = lfp_file.create_dataset('MinVal', data = np.amin(lfpData))
	maxVal = lfp_file.create_dataset('MaxVal', data = np.amax(lfpData))
	highFreqCorner = lfp_file.create_dataset('HighFreqCorner', data = lowPassFrequency[0]*resampleRate)
	lowFreqCorner = lfp_file.create_dataset('LowFreqCorner', data = lowPassFrequency[1]*resampleRate)
	numberSamples = lfp_file.create_dataset('NumberSamples', data = len(lfpData))
	highFreqOrder = lfp_file.create_dataset('HighFreqOrder', data = LFPOrder)
	lowFreqOrder = lfp_file.create_dataset('LowFreqOrder', data = LFPOrder)
	probeInfo = lfp_file.create_dataset('ProbeInfo', data = str(np.array(data['ProbeInfo'])).replace('raw', 'lfp'))
	data.close()
	lfp_file.close()
	print("rpllfp.hdf5 has been written")
	return 

rpllfp()