import numpy as np 
from . import lowPassFilter 
from DataProcessingTools import DPT 

class RPLLFP(DPT.DPObject):
	def __init__(self):
		DPT.DPObject.__init__(self)
		self.analogData = []
		self.analogInfo = {}

	def plot():
		pass 

def rpllfp(saveLevel = 0, redoLevel = 0, data = [], lowPassFrequency = [1, 150], LFPOrder = 6, resampleRate = 1000, display = False, saveFig = False):
	if len(data) > 0: 
		analogData = data 
		# There isnt a data object so how does this work now? 
	else:
		rw = RPLRaw(saveLevel = saveLevel - 1)
		rw.load('rplraw.hkl')
		if not rw.isempty(): 
			analogData = rw.analogData 
			analogInfo = rw.analogInfo 
			samplingRate = analogInfo['SampleRate']
			lfpData, resampleRate = lowPassFilter(analogData, samplingRate = samplingRate, resampleRate = resampleRate, LFPOrder = LFPOrder, lowFreq = lowPassFrequency[0], highFreq = lowPassFrequency[1], display = False, saveFig = False)
			analogInfo['SampleRate'] = resampleRate
			analogInfo['MinVal'] = np.amin(lfpData)
			analogInfo['MaxVal'] = np.amax(lfpData)
			analogInfo['HighFreqCorner'] = lowPassFrequency[0] * resampleRate
			analogInfo['LowFreqCorner'] = lowPassFrequency[1] * resampleRate
			analogInfo['NumberSamples'] = len(lfpData)
			analogInfo['HighFreqOrder'] = LFPOrder
			analogInfo['LowFreqOrder'] = LFPOrder
			analogInfo['ProbeInfo'] = analogInfo['ProbeInfo'].replace('raw', 'lfp')
			rlfp = RPLLFP()
			rlfp.data = lfpData
			rlfp.analogInfo = analogInfo
		else: 
			rlfp = RPLLFP()
		if saveLevel > 0: 
			rlfp.save()
		return 

