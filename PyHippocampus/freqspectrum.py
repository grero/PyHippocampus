import numpy as np 
from scipy import signal 
import DataProcessingTools as DPT 
from .rpllfp import RPLLFP 
from .rplhighpass import RPLHighPass
from .rplraw import RPLRaw
from .helperfunctions import plotFFT
from matplotlib.pyplot import gca
import os 

class FreqSpectrum(DPT.DPObject):

	filename = "freqspectrum.hkl"
	argsList = [('loadHighPass', False), ('loadRaw', False)]
	level = 'channel'

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.freq = []
		self.magnitude = []
		self.numSets = 0 
		if self.args['loadHighPass']:
			rpdata = RPLHighPass()
		elif self.args['loadRaw']:
			rpdata = RPLRaw()
		else: 
			rpdata = RPLLFP()
		if len(rpdata.data) > 0: 
			DPT.DPObject.create(self, *args, **kwargs)
			self.magnitude, self.freq = plotFFT(rpdata.data, rpdata.analogInfo['SampleRate'])
			self.freq = [list(self.freq)]
			self.magnitude = [list(self.magnitude)]
			self.numSets = 1
			self.title = [DPT.levels.get_shortname("channel", os.getcwd())[-3:]]
		else: 
			DPT.DPObject.create(self, dirs=[], *args, **kwargs)   
		return 

	def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):

		plotOpts = {'LabelsOff': False, 'Type': DPT.objects.ExclusiveOptions(['channel', 'array'], 0), 'TitleOff': False, 'XLims': []}

		if getPlotOpts:
			return plotOpts

		if getLevels: 
			return ['channel', 'array']

		plot_type = plotOpts["Type"].selected()
		if getNumEvents:
			if plot_type == 'channel':
				if i == None: 
					idx = 0 
				else: 
					idx = i 
				return len(self.magnitude), idx

		if ax is None:
			ax = plt.gca()

		if not overlay:
			ax.clear()

		if plot_type == 'channel':
			y = self.magnitude[i]
			x = self.freq[i]
			ax.plot(x, y)
			if not plotOpts['LabelsOff']:
				ax.set_xlabel('Freq (Hz)')
				ax.set_ylabel('Magnitude')
			if not plotOpts['TitleOff']:
				ax.set_title('channel' + str(self.title[i]))
			if len(plotOpts['XLims']) > 0: 
				ax.set_xlim(plotOpts['XLims'])
			else: 
				if self.args['loadHighPass']:
					ax.set_xlim([500, 7500])
				elif self.args['loadRaw']:
					ax.set_xlim([0, 10000])
				else:
					ax.set_xlim([0, 150])
		elif plot_type == 'array':
			pass 
		return ax

	def append(self, fs):
		DPT.DPObject.append(self, fs)
		self.numSets += fs.numSets
		self.magnitude += [fs.magnitude]
		self.freq += [fs.freq]
		self.title += [fs.title]
