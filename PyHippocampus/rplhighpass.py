from .rplraw import RPLRaw 
from scipy import signal 
import numpy as np 
import DataProcessingTools as DPT 
from .helperfunctions import plotFFT

def highPassFilter(analogData, samplingRate = 30000, lowFreq = 500, highFreq = 7500, HPOrder = 4):
	analogData = analogData.flatten()
	fn = samplingRate / 2
	lowFreq = lowFreq / fn 
	highFreq = highFreq / fn 
	[b, a] = signal.butter(HPOrder, [lowFreq, highFreq], 'bandpass')
	print("Applying high-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
	hps = signal.filtfilt(b, a, analogData, padtype = 'odd', padlen = 3*(max(len(b),len(a))-1))
	return hps, samplingRate

class RPLHighPass(DPT.DPObject):

	filename = "rplhighpass.hkl"
	argsList = [("HighOrder", 8), ("HighPassFrequency", [500, 7500])]
	level = "channel"

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = []
		self.analogInfo = {}
		self.numSets = 0
		rw = RPLRaw()
		if len(rw.data) > 0: 
			hpData, samplingRate = highPassFilter(rw.data, samplingRate = rw.analogInfo['SampleRate'], HPOrder = int(self.args['HighOrder'] / 2), lowFreq = self.args['HighPassFrequency'][0], highFreq = self.args['HighPassFrequency'][1])
			self.analogInfo['SampleRate'] = samplingRate
			self.analogInfo['MinVal'] = np.amin(hpData)
			self.analogInfo['MaxVal'] = np.amax(hpData)
			self.analogInfo['HighFreqCorner'] = self.args['HighPassFrequency'][0] * samplingRate
			self.analogInfo['LowFreqCorner'] = self.args['HighPassFrequency'][1] * samplingRate
			self.analogInfo['NumberSamples'] = len(hpData)
			self.analogInfo['HighFreqOrder'] = self.args['HighOrder']
			self.analogInfo['LowFreqOrder'] = self.args['HighOrder']
			self.analogInfo['ProbeInfo'] = rw.analogInfo['ProbeInfo'].replace('raw', 'hp')
			self.data = hpData
			self.numSets = 1 
		return self

		def plot(self, i = None, ax = None, overlay = False):
			self.current_idx = i 
			if ax is None: 
				ax = plt.gca()
			if not overlay:
				ax.clear()
			self.plotopts = {'LabelsOff': False, 'GroupPlots': 1, 'GroupPlotIndex': 1, 'Color': 'b', 'FFT': False, 'XLims': [0, 150], 'LoadSort': False}
			plot_type_FFT = self.plotopts['FFT']
			if plot_type_FFT: 
				ax = plotFFT(self.data, self.analogInfo['SampleRate'])
				if not self.plotopts['LabelsOff']:
					ax.set_xlabel('Freq (Hz)')
					ax.set_ylabel('Magnitude')
				ax.xlim(self.plotopts['XLims'])
			else:
				ax.plot(self.data)
				if self.plotopts['LoadSort']:
					pass  
			return 


