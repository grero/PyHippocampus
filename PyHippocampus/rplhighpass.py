import numpy as np 
from . import highPassFilter 
import DataProcessingTools as DPT 

class RPLHighPass(DPT.DPObject):
	def __init__(self):
		DPT.DPObject().__init__(self)
		self.analogData = []
		self.analogInfo = {}

	def plot():
		pass 

def rplhighpass(saveLevel = 0, redoLevel = 0, highPassFrequency = [500, 7500], highOrder = 6):
	rw = RPLRaw()
	rw.load('rplraw.hkl')
	analogData = rw.analogData
	analogInfo = rw.analogInfo
	hpData, samplingRate = highPassFilter(analogData, samplingRate = samplingRate, HPOrder = highOrder, lowFreq = highPassFrequency[0], highFreq = highPassFrequency[1])
	analogInfo['SampleRate'] = samplingRate
	analogInfo['MinVal'] = np.amin(hpData)
	analogInfo['MaxVal'] = np.amax(hpData)
	analogInfo['HighFreqCorner'] = highPassFrequency[0] * samplingRate
	analogInfo['LowFreqCorner'] = highPassFrequency[1] * samplingRate
	analogInfo['NumberSamples'] = len(hpData)
	analogInfo['HighFreqOrder'] = highOrder
	analogInfo['LowFreqOrder'] = highOrder
	analogInfo['ProbeInfo'] = analogInfo['ProbeInfo'].replace('raw', 'hp')
	hp = RPLHighPass()
	hp.analogData = analogData
	hp.analogInfo = analogInfo
	if saveLevel > 0:
		hp.save()
	return 


# def rplhighpass(saveLevel = 0, redoLevel = 0, highPassFrequency = [500, 7500], highOrder = 8):
# 	file = 'rplraw.hdf5'
# 	data = h5.File(file, 'r')
# 	analogData = np.array(data['analogSignal'])
# 	samplingRate = np.array(data['SampleRate'])
# 	hpData, samplingRate = highPassFilter(analogData, samplingRate = samplingRate, HPOrder = highOrder, lowFreq = highPassFrequency[0], highFreq = highPassFrequency[1])
# 	highpass_file = h5.File('rplhighpass.hdf5', 'w')
# 	hAnalogData = highpass_file.create_dataset('analogData', data = hpData)
# 	sampleRate = highpass_file.create_dataset('SampleRate', data = samplingRate)
# 	minVal = highpass_file.create_dataset('MinVal', data = np.amin(hpData))
# 	maxVal = highpass_file.create_dataset('MaxVal', data = np.amax(hpData))
# 	highFreqCorner = highpass_file.create_dataset('HighFreqCorner', data = highPassFrequency[0]*samplingRate)
# 	lowFreqCorner = highpass_file.create_dataset('LowFreqCorner', data = highPassFrequency[1]*samplingRate)
# 	numberSamples = highpass_file.create_dataset('NumberSamples', data = len(hpData))
# 	highFreqOrder = highpass_file.create_dataset('HighFreqOrder', data = highOrder)
# 	lowFreqOrder = highpass_file.create_dataset('LowFreqOrder', data = highOrder)
# 	probeInfo = highpass_file.create_dataset('ProbeInfo', data = str(np.array(data['ProbeInfo'])).replace('raw', 'hp'))
# 	data.close()
# 	highpass_file.close()
# 	print("rplhighpass.hdf5 has been written")
# 	return 
