import numpy as np 
from scipy import signal 
import DataProcessingTools as DPT 
from .rplraw import RPLRaw

def resampleData(analogData, samplingRate, resampleRate):
	numberOfPoints = int(resampleRate * (len(analogData) / samplingRate))
	analogData = signal.resample(analogData, numberOfPoints)
	return analogData

def lowPassFilter(analogData, samplingRate = 30000, resampleRate = 1000, lowFreq = 1, highFreq = 150, LFPOrder = 4, padlen = 0):
	analogData = analogData.flatten()
	lfpsData = resampleData(analogData, samplingRate, resampleRate)
	fn = resampleRate / 2
	lowFreq = lowFreq / fn 
	highFreq = highFreq / fn 
	[b, a] = signal.butter(LFPOrder, [lowFreq, highFreq], 'bandpass')
	print("Applying low-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
	lfps = signal.filtfilt(b, a, lfpsData, padlen = padlen)
	return lfps, resampleRate

class RPLLFP(DPT.DPObject):

	filename = "rpllfp.hkl"
	argsList = [('SampleRate', 30000), ('ResampleRate', 1000), ('LFPOrder', 4), ('LowPassFrequency', [1, 150])]
	level = 'channel'

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = []
		self.analogInfo = {}
		self.numSets = 0 
		rw = RPLRaw()
		lfpData, resampleRate = lowPassFilter(rw.data, samplingRate = self.args['SampleRate'], resampleRate = self.args['ResampleRate'], LFPOrder = self.args['LFPOrder'], lowFreq = self.args['LowPassFrequency'][0], highFreq = self.args['LowPassFrequency'][1])
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
		
	def plot(self, i = None, ax = None, overlay = False):
		self.current_idx = i 
		if ax is None: 
			ax = plt.gca()
		if not overlay:
			ax.clear()
		self.plotopts = {'LabelsOff': False, 'GroupPlots': 1, 'GroupPlotIndex': 1, 'Color': 'b', 'FFT': False, 'XLims': [0, 150]}
		plot_type_FFT = self.plotopts['FFT']
		if plot_type_FFT: 
			# TODO: Complete PlotFFT Function. 
			ax = PlotFFT(self.data, self.analogInfo['SampleRate'])
			if not self.plotopts['LabelsOff']:
				ax.set_xlabel('Freq (Hz)')
				ax.set_ylabel('Magnitude')
		else:
			ax.plot(self.data)
			if not self.plotopts['LabelsOff']:
				ax.set_ylabel('Voltage (uV)')
				ax.set_xlabel('Time (ms)')
		direct = os.getcwd()
		session = DPT.levels.get_shortname("session", direct)
		array = DPT.levels.get_shortname("array", direct)
		channel = DPT.levels.get_shortname("channel", direct)
		title = session + array + channel
		ax.set_title(title)
		return ax 
