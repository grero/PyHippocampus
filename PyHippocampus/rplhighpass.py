from . import RPLRaw 
from scipy import signal 
import numpy as np 
import DataProcessingTools as DPT 

def highPassFilter(analogData, samplingRate = 30000, lowFreq = 500, highFreq = 7500, HPOrder = 8, padlen = 0):
	analogData = analogData.flatten()
	fn = samplingRate / 2
	lowFreq = lowFreq / fn 
	highFreq = highFreq / fn 
	sos = signal.butter(HPOrder, [lowFreq, highFreq], 'bandpass', fs = samplingRate, output = "sos")
	print("Applying high-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
	hps = signal.sosfiltfilt(sos, analogData, padlen = padlen)
	return hps, samplingRate

class RPLHighPass(DPT.DPObject):

	filename = "rplhighpass.hkl"
	argsList = [("HighOrder", 6), ("HighPassFrequency", [500, 7500])]
	level = "channel"

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = []
		self.analogInfo = {}
		rw = RPLRaw()
		hpData, samplingRate = highPassFilter(rw.data, samplingRate = rw.analogInfo['SampleRate'], HPOrder = self.args['HighOrder'], lowFreq = self.args['HighPassFrequency'][0], highFreq = self.args['HighPassFrequency'][1])
		self.analogInfo['SampleRate'] = samplingRate
		self.analogInfo['MinVal'] = np.amin(hpData)
		self.analogInfo['MaxVal'] = np.amax(hpData)
		self.analogInfo['HighFreqCorner'] = self.args['HighPassFrequency'][0] * samplingRate
		self.analogInfo['LowFreqCorner'] = self.args['HighPassFrequency'][1] * samplingRate
		self.analogInfo['NumberSamples'] = len(hpData)
		self.analogInfo['HighFreqOrder'] = self.args['HPOrder']
		self.analogInfo['LowFreqOrder'] = self.args['HPOrder']
		self.analogInfo['ProbeInfo'] = rw.analogInfo['ProbeInfo'].replace('raw', 'hp')
		self.data = hpData
		return self

		def plot(self):
			pass 
