import numpy as np 
import DataProcessingTools as DPT 
from .rplparallel import RPLParallel
from .rpllfp import RPLLFP
from .helperfunctions import plotFFT
import os 

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

		plotOpts = {'LabelsOff': False, 'PreTrial': 500, 'NormalizeTrial': False, 'RewardMarker': 3, 'TimeOutMarker': 4, 'PlotAllData': False, 'TitleOff': False, 'RemoveLineNoise': [], 'LogPlot': False, 'FreqLims': [], 'TFfttWindow': 200, 'TFfttOverlap': 150, 'TFfftPoints': 256, 'TFfftStart': 500, 'TFfftFreq': 150, 'TimeWindow': [], 'FilterWindow': [], "Type": DPT.objects.ExclusiveOptions(["FreqPlot", 'Signal'], 1)} 
		# 'CorrCoeff', 'TFfft', 'TFWavelets', 'Filter', 'OverlapLFP'
		# Add flag for removelinenoise and a specific value. 

		plot_type = plotOpts['Type'].selected()

		if getPlotOpts:
			return plotOpts 

		if getLevels:
			return ['trial', 'all']

		if getNumEvents:
			if plotOpts['PlotAllData']: # to avoid replotting the same data. 
				return 1, 0 
			if plot_type == 'FreqPlot' or 'Signal' or 'TFfft':
				if i is not None:
					nidx = i 
				else:
					nidx = 0
				return self.numSets, nidx 

		if ax is None:
			ax = plt.gca()

		if not overlay:
			ax.clear()
		
		# Magintude of the complex number and square it -> power density. 

		sRate = self.samplingRate
		trialIndicesForN = self.trialIndices[i, :] 
		idx = [int(trialIndicesForN[0] - ((plotOpts['PreTrial'] / 1000) * sRate))] + list(trialIndicesForN[1:])

		self.analogTime = [i / sRate for i in range(len(self.data))]

		if plot_type == 'Signal':
			data = self.data[idx[0]:idx[-1]]
			if len(plotOpts['RemoveLineNoise']) > 0:
				data = removeLineNoise(data, plotOpts['removeLineNoise'], sRate)
			x = np.linspace(-plotOpts['PreTrial'], 0, num = plotOpts['PreTrial'])
			x = np.concatenate((x, np.linspace(0, len(data) - plotOpts['PreTrial'], num = len(data) - plotOpts['PreTrial'])))
			# x = np.array(self.analogTime[idx[0]:idx[-1]])
			# print(len(x), len(data))
			# print(type(x), type(data))
			ax.plot(x, data)
			ax.axvline(0, color = 'g') # Start of trial. 
			ax.axvline((self.timeStamps[i][1] - self.timeStamps[i][0]) * 30000, color = 'm')
			ax.axvline((self.timeStamps[i][2] - self.timeStamps[i][0]) * 30000, color = 'r')

		elif plot_type == 'FreqPlot':
			if plotOpts['PlotAllData']:
				data = self.data 
			else: 
				data = self.data[idx[0]:idx[-1]]
			if len(plotOpts['RemoveLineNoise']) > 0:
				data = removeLineNoise(data, plotOpts['removeLineNoise'], sRate)
			datam = np.mean(data)
			fftProcessed, f = plotFFT(data - datam, sRate)
			ax.plot(f, fftProcessed)
			if plotOpts['LogPlot']:
				ax.set_yscale('log')

		elif plot_type == 'TFftt': 
			tIdx = self.trialIndices[i,:]
			idx = [tIdx[0] - ((plotOpts['TFfftStart']+500)/1000*sRate), tIdx[0] - ((plotOpts['TFfftStart']+1)/1000*sRate)]
			data = self.data[idx[0]:idx[-1]]
			datam = np.mean(data)
			pass 
            # data = obj.data.analogData(idx);
            # datam = mean(data);
            # [~,~,~,P]=spectrogram(data-datam,Args.TFfftWindow,Args.TFfftOverlap,Args.TFfftPoints,sRate,'yaxis');
            
            # %     Normalization parameters of the NP
            # Pmean=mean(P,2); %mean power density of each frequency bin
            # Pstd=std(P,0,2); %standard deviation of each frequency bin

		elif plot_type == 'CorrCoeff':
			pass 

		elif plot_type == 'TFWavelets':
			pass 

		elif plot_type == 'Filter':
			pass 

		elif plot_type == 'OverlapLFP':
			pass 

		if not plotOpts['LabelsOff']:
			if plot_type == 'FreqPlot':
				ax.set_xlabel('Frequency (Hz)')
				ax.set_ylabel('Magnitude')
			elif plot_type == 'TFfft':
				ax.set_xlabel('Time (s)')
				ax.set_ylabel('Frequency (Hz)')
			else:
				ax.set_xlabel('Time (ms)')
				ax.set_ylabel('Voltage (uV)')

		if not plotOpts['TitleOff']:
			channel = DPT.levels.get_shortname("channel", os.getcwd())[1:]
			ax.set_title('channel' + str(channel))

		if len(plotOpts['FreqLims']) > 0:
			if plot_type == 'FreqPlot':
				ax.xlim(plotOpts['FreqLims'])
			elif plot_type == 'TFfft':
				ax.ylim(plotOpts['FreqLims'])
		return ax 






