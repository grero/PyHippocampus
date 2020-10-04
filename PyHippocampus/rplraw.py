import numpy as np 
import DataProcessingTools as DPT
from matplotlib.pyplot import gca
from .helperfunctions import computeFFT
import os
from . import rplsplit

class RPLRaw(DPT.DPObject):

	filename = 'rplraw.hkl'
	argsList = []
	level = 'channel'

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = np.array([])
		self.analogInfo = {}
		if 'analogData' in kwargs.keys() and 'analogInfo' in kwargs.keys():
			# create object
			DPT.DPObject.create(self, *args, **kwargs)
			self.data = kwargs['analogData'].flatten()
			self.analogInfo = kwargs['analogInfo']
			self.numSets = 1
		else:
			# check if in channel directory, if so create rplraw object by calling rplsplit. 
			if DPT.levels.level(os.getcwd()) == 'channel':
				channelNumber = int(DPT.levels.get_level_name('channel', os.getcwd())[-3:])
				rs = rplsplit.RPLSplit(returnData = True, channel = [channelNumber])
    			# create object
				DPT.DPObject.create(self, *args, **kwargs)
				self.data = rs.data.flatten() 
				self.analogInfo = rs.analogInfo
				self.numSets = 1 
			# create empty object
			else: 
				DPT.DPObject.create(self, dirs=[], *args, **kwargs)            
		return self
	
	def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):

		plotOpts = {'LabelsOff': False, 'FFT': False, 'XLims': [0, 150], 'TimeSplit': 10, 'PlotAllData': False}

		for (k, v) in plotOpts.items():
			plotOpts[k] = kwargs.get(k, v)

		if getPlotOpts:
			return plotOpts

		if getNumEvents:
			# Return the number of events avilable
			if plotOpts['FFT'] or plotOpts['PlotAllData']:
				return 1, 0 
			else:
				if i is not None:
					idx = i 
				else:
					idx = 0 
				totalEvents = len(self.data) / (self.analogInfo['SampleRate'] * plotOpts['TimeSplit'])
				return totalEvents, i

		if getLevels:        
			# Return the possible levels for this object
			return ["channel", 'trial']

		if ax is None:
			ax = gca()
		if not overlay:
			ax.clear()

		self.analogTime = [(i * 1000) / self.analogInfo["SampleRate"] for i in range(len(self.data))]
	
		plot_type_FFT = plotOpts['FFT']
		if plot_type_FFT: 
			fftProcessed, f = plotFFT(self.data, self.analogInfo['SampleRate'])
			ax.plot(f, fftProcessed)
			if not plotOpts['LabelsOff']:
				ax.set_xlabel('Freq (Hz)')
				ax.set_ylabel('Magnitude')
			ax.set_xlim(plotOpts['XLims'])
		else:
			if plotOpts['PlotAllData']:
				ax.plot(self.analogTime, self.data)
			else: 
				idx = [self.analogInfo['SampleRate'] * plotOpts['TimeSplit'] * i, self.analogInfo['SampleRate'] * plotOpts['TimeSplit'] * (i + 1) + 1] 
				data = self.data[int(idx[0]):int(idx[1])]
				time = self.analogTime[int(idx[0]):int(idx[1])] 
				ax.plot(time, data)
			if not plotOpts['LabelsOff']:
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
