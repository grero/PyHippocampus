import numpy as np 
import DataProcessingTools as DPT 
from .rplparallel import RPLParallel
from .rpllfp import RPLLFP
from .helperfunctions import plotFFT

class VMLFP(DPT.DPObject):

	filename = 'vmflp.hkl'
	argsList = []
	level = 'channel'

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.markers = np.array([])
		self.timeStamps = np.array([])
		self.trialIndices = np.array([])
		self.data = np.array([])
		self.samplingRate = None 
		self.numSets = 0  
		rp = RPLParallel()
		rlfp = RPLLFP()
		if len(rlfp.data) > 0 and len(rp.timeStamps) > 0: 
			self.data = rlfp.data 
			self.markers = rp.markers 
			self.samplingRate = rlfp.analogInfo['SampleRate']
			dsample = rp.samplingRate / rlfp.analogInfo['SampleRate']
			self.timeStamps = rp.timeStamps / dsample 
			self.trialIndices = np.rint(rp.trialIndices / dsample).astype('int')
			self.numSets = self.trialIndices.shape[0]
		return self 

	def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):

		self.current_idx = i
		if ax is None:
			ax = plt.gca()
		if not overlay:
			ax.clear()

		plotops = {'LabelsOff': False, 'PreTrial': 500, 'NormalizeTrial': False, 'RewardMarker': 3, 'TimeOutMarker': 4, 'PlotAllData': False, 'TitleOff': False, 'RemoveLineNoise': [], 'LogPlot': False, 'FreqLims': [], 'TFfttWindow': 200, 'TFfttOverlap': 150, 'TFfftPoints': 256, 'TFfftStart': 500, 'TFfftFreq': 150, 'TimeWindow': [], 'FilterWindow': [], "Type": DPT.objects.ExclusiveOptions(['CorrCoeff', "FreqPlot", 'TFfft', 'TFWavelets', 'Filter', 'OverlapLFP'], 0)} 
		plot_type = self.plotops['Type'].selected()

		trialIndicesForN = self.trialIndices[n, :] 
		sRate = self.analogInfo['SampleRate'] 
		idx = trialIndicesForN[0] - (preTrial / 1000 * sRate) + trialIndicesForN[1:]

		if plot_type == "FreqPlot":
			if self.plotopts['plotAllData']:
				data = data
			else:
				data = data[idx]
			if len(removeLineNoise) > 0: 
				data = nptRemoveLineNoise(data, RemoveLineNoise, sRate)
			datam = np.mean(data)
			ax = plotFFT(data - datam, sRate)
			if self.plotopts['logPlot']:
				ax.set_yscale('log')

		elif plot_type == 'TFftt': 
			if self.plotopts['plotAllData']:
				dIdx = self.trialIndices[:, -1] - self.trialIndices[:, 0]
				mIdx = np.amax(dIdx)
				spTimeStep = self.plotopts['TFfttWindow'] - self.plotopts['TFfttOverlap'] 
				spTimeBins = np.floor(mIdx/spTimeStep) - self.plotopts['TFfttOverlap']/spTimeStep
				nFreqs = (self.plotopts['TFfttPoints']/2) + 1 
				ops = np.zeroes([nFreqs, spTimeBins])
				opsCount = ops
				for i in range(numSets):
					tftIdx = self.trialIndices[i, :]
					tfidx = tftIdx[0:]
					data = data[tfidx]
					if len(removeLineNoise) > 0: 
						data = nptremoveLineNoise(data, removeLineNoise, sRate)
					datam = np.mean(data)
					pass 
			else: 	
				pass 

		elif plot_type == 'CorrCoeff':
			pass 

		elif plot_type == 'TFWavelets':
			pass 

		elif plot_type == 'Filter':
			pass 

		elif plot_type == 'OverlapLFP':
			pass 

		else:
			pass 

		if not self.plotopts['LabelsOff']:
			if plot_type == 'FreqPlot':
				ax.set_xlabel('Frequency (Hz)')
				ax.set_ylabel('Magnitude')
			elif plot_type == 'TFfft':
				ax.set_xlabel('Time (s)')
				ax.set_ylabel('Frequency (Hz)')
			else:
				ax.set_xlabel('Time (ms)')
				ax.set_ylabel('Voltage (uV)')

		if not self.plotopts['TitleOff']:
			ax.set_title('hehe') # Fix this. 

		if len(self.plotopts['FreqLims']) > 0:
			if plot_type == 'FreqPlot':
				ax.xlim(self.plotopts['FreqLims'])
			elif plot_type == 'TFfft':
				ax.ylim(self.plotopts['FreqLims'])
		return ax 






