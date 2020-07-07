import numpy as np 
from scipy import signal 
import DataProcessingTools as DPT 
from rplraw import RPLRaw
import PanGUI

def resampleData(analogData, samplingRate, resampleRate):
	numberOfPoints = int(resampleRate * (len(analogData) / samplingRate))
	analogData = signal.resample(analogData, numberOfPoints)
	return analogData

def lowPassFilter(analogData, samplingRate = 30000, resampleRate = 1000, lowFreq = 1, highFreq = 150, LFPOrder = 8, padlen = 0, display = False, saveFig = False):
	analogData = analogData.flatten()
	lfpsData = resampleData(analogData, samplingRate, resampleRate)
	fn = resampleRate / 2
	lowFreq = lowFreq / fn 
	highFreq = highFreq / fn 
	sos = signal.butter(LFPOrder, [lowFreq, highFreq], 'bandpass', fs = resampleRate, output = "sos")
	print("Applying low-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
	lfps = signal.sosfiltfilt(sos, lfpsData, padlen = padlen)
	if display: 
		lfpPlot(analogData, lfpsData, lfps, saveFig = saveFig)
		print('saved figure')
	return lfps, resampleRate

class RPLLFP(DPT.DPObject):

	filename = 'rpllfp.hkl'
	argsList = [('sampleRate', 30000), ('resampleRate', 1000), ('LFPOrder', 6), ('lowPassFrequency', [1, 150])]

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, normpath = False, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = []
		self.analogInfo = {}
		rw = RPLRaw()
		analogData = rw.data 
		LFPOrder = self.args['LFPOrder']
		lfpData, resampleRate = lowPassFilter(analogData, samplingRate = self.args['sampleRate'], resampleRate = self.args['resampleRate'], LFPOrder = self.args['LFPOrder'], lowFreq = self.args['lowPassFrequency'][0], highFreq = self.args['lowPassFrequency'][1])
		self.analogInfo['SampleRate'] = resampleRate
		self.analogInfo['MinVal'] = np.amin(lfpData)
		self.analogInfo['MaxVal'] = np.amax(lfpData)
		self.analogInfo['HighFreqCorner'] = self.args['lowPassFrequency'][0] * resampleRate
		self.analogInfo['LowFreqCorner'] = self.args['lowPassFrequency'][1] * resampleRate
		self.analogInfo['NumberSamples'] = len(lfpData)
		self.analogInfo['HighFreqOrder'] = self.args['LFPOrder']
		self.analogInfo['LowFreqOrder'] = self.args['LFPOrder']
		self.analogInfo['ProbeInfo'] = rw.analogInfo['ProbeInfo'].replace('raw', 'lfp')
		if kwargs.get('saveLevel', 0) > 0:
			self.save()
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
