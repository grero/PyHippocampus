import numpy as np 
from scipy import signal 
import DataProcessingTools as DPT 
from .rplraw import RPLRaw
from .helperfunctions import plotFFT
import matplotlib.pyplot as plt
import os 

def lowPassFilter(analogData, samplingRate = 30000, resampleRate = 1000, lowFreq = 1, highFreq = 150, LFPOrder = 4):
	analogData = analogData.flatten()
	lfpsData = signal.resample_poly(analogData, resampleRate, samplingRate)
	fn = resampleRate / 2
	lowFreq = lowFreq / fn 
	highFreq = highFreq / fn 
	[b, a] = signal.butter(LFPOrder, [lowFreq, highFreq], 'bandpass')
	print("Applying low-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
	lfps = signal.filtfilt(b, a, lfpsData, padtype = 'odd', padlen = 3*(max(len(b),len(a))-1))
	return lfps, resampleRate

class RPLLFP(DPT.DPObject):

	filename = "rpllfp.hkl"
	argsList = [('SampleRate', 30000), ('ResampleRate', 1000), ('LFPOrder', 8), ('LowPassFrequency', [1, 150])]
	level = 'channel'

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = []
		self.analogInfo = {}
		self.numSets = 0 
		rw = RPLRaw()
		if len(rw.data) > 0:
			lfpData, resampleRate = lowPassFilter(rw.data, samplingRate = self.args['SampleRate'], resampleRate = self.args['ResampleRate'], LFPOrder = int(self.args['LFPOrder'] / 2), lowFreq = self.args['LowPassFrequency'][0], highFreq = self.args['LowPassFrequency'][1])
			self.analogInfo['SampleRate'] = resampleRate
			self.analogInfo['MinVal'] = np.amin(lfpData)
			self.analogInfo['MaxVal'] = np.amax(lfpData)
			self.analogInfo['HighFreqCorner'] = self.args['LowPassFrequency'][0] * resampleRate
			self.analogInfo['LowFreqCorner'] = self.args['LowPassFrequency'][1] * resampleRate
			self.analogInfo['NumberSamples'] = len(lfpData)
			self.analogInfo['HighFreqOrder'] = self.args['LFPOrder']
			self.analogInfo['LowFreqOrder'] = self.args['LFPOrder']
			self.analogInfo['ProbeInfo'] = rw.analogInfo['ProbeInfo'].replace('raw', 'lfp')
			self.data = lfpData
			self.numSets = 1 
		return self 
		
	def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):
		self.current_idx = i 
		if ax is None: 
			ax = plt.gca()
		if not overlay:
			ax.clear()

		self.plotopts = {'LabelsOff': False, 'Color': 'b', 'FFT': False, 'XLims': [0, 150], 'level': 'channel'}

		for (k, v) in self.plotopts.items():
			self.plotopts[k] = kwargs.get(k, v)

		if getPlotOpts:
			return self.plotopts

		if getNumEvents:
			# Return the number of events avilable
			return i, 1 
		if getLevels:        
			# Return the possible levels for this object
			return ["channel"]
	
		plot_type_FFT = self.plotopts['FFT']
		if plot_type_FFT: 
			fftProcessed, f = plotFFT(self.data, self.analogInfo['SampleRate'])
			ax.plot(f, fftProcessed)
			if not self.plotopts['LabelsOff']:
				ax.set_xlabel('Freq (Hz)')
				ax.set_ylabel('Magnitude')
			ax.set_xlim(self.plotopts['XLims'])
		else:
			ax.plot(self.data)
			if not self.plotopts['LabelsOff']:
				ax.set_ylabel('Voltage (uV)')
				ax.set_xlabel('Time (ms)')
		direct = os.getcwd()
		day = DPT.levels.get_shortname('day', direct)
		session = DPT.levels.get_shortname("session", direct)
		array = DPT.levels.get_shortname("array", direct)
		channel = DPT.levels.get_shortname("channel", direct)
		title = 'D' + day + session + array + channel
		ax.set_title(title)
		return ax 
