import numpy as np 
from scipy import signal 
import DataProcessingTools as DPT 
from rplraw import RPLRaw

def highPassFilter(analogData, samplingRate = 30000, lowFreq = 500, highFreq = 7500, HPOrder = 8, padlen = 0, display = False, savefig = False):
	analogData = analogData.flatten()
	fn = samplingRate / 2
	lowFreq = lowFreq / fn 
	highFreq = highFreq / fn 
	sos = signal.butter(HPOrder, [lowFreq, highFreq], 'bandpass', fs = samplingRate, output = "sos")
	print("Applying high-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
	hps = signal.sosfiltfilt(sos, analogData, padlen = padlen)
	if display: 
		highpassPlot(analogData, hps, lowFreq = lowFreq * fn, highFreq = highFreq * fn, saveFig = False)
	return hps, samplingRate

class RPLHighPass(DPT.DPObject):

	filename = 'rplhighpass.hkl'
	argsList = [('highOrder', 6), ('highPassFrequency', [500, 7500])]

	def __init__(self, *args, **kwargs):
		DPT.DPObject().__init__(self, normpath = False, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = []
		self.analogInfo = {}
		rw = RPLRaw()
		analogData = rw.data 
		hpData, samplingRate = highPassFilter(analogData, samplingRate = rw.analogInfo['sampleRate'], HPOrder = self.args['highOrder'], lowFreq = self.args['highPassFrequency'][0], highFreq = self.args['highPassFrequency'][1])
		self.analogInfo['SampleRate'] = samplingRate
		self.analogInfo['MinVal'] = np.amin(hpData)
		self.analogInfo['MaxVal'] = np.amax(hpData)
		self.analogInfo['HighFreqCorner'] = self.args['highPassFrequency'][0] * samplingRate
		self.analogInfo['LowFreqCorner'] = self.args['highPassFrequency'][1] * samplingRate
		self.analogInfo['NumberSamples'] = len(hpData)
		self.analogInfo['HighFreqOrder'] = self.args['highOrder']
		self.analogInfo['LowFreqOrder'] = self.args['highOrder']
		self.analogInfo['ProbeInfo'] = rw.analogInfo['ProbeInfo'].replace('raw', 'hp')
		if kwargs.get('saveLevel', 0) > 0:
			self.save
		return self
		
	def plot(self, i = None, ax = None, overlay = False):
		self.current_idx = i 
		if ax is None: 
			ax = plt.gca()
		if not overlay:
			ax.clear()
		self.plotopts = {'LabelsOff': False, 'GroupPlots': 1, 'GroupPlotIndex': 1, 'Color': 'b', 'FFT': False, 'XLims': [0, 10000], 'SpikeData': [], 'SpikeTriggerIndex': 26, 'SpikeHeight': 100, 'LoadSort': False}
		plot_type_FFT = self.plotopts['FFT']
		if plot_type_FFT:
			ax = PlotFFT(self.data, self.analogInfo['SampleRate']) # TODO 
			if not self.plotopts['LabelsOff']:
				ax.set_xlabel('Freq (Hz)')
				ax.set_ylabel('Magnitude')
			ax.xlim(self.plotopts['XLims'])
		else:
			pass 
		return ax 


