import neo  
from neo.io import BlackrockIO 
import numpy as np 
import os 
import DataProcessingTools as DPT 

# RPLSplit to give the data to rplraw. 
# If no data, rplraw to call rplsplit to open and pass it the data. 

class RPLRaw(DPT.DPObject):

	filename = 'rplraw.hkl'
	argsList = [('Data', True), ('analogInfo', {}), ('analogData', [])]

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = []
		self.analogInfo = {}

		self.data = self.args['analogData']
		self.analogInfo = self.args['analogInfo']
		try: # If no data is presented, generates empty object. 
			arrayNumber = self.analogInfo['ArrayNumber']
			channelNumber = self.analogInfo['ChannelNumber']
		except: 
			continue 
		self.numSets = 1

		if kwargs.get('saveLevel', 0) > 0:
			directory = os.getcwd() # Go to the channel directory and save file there. 
			if arrayNumber not in os.listdir('.'): 
				os.mkdir(arrayNumber)
			path = os.path.join(directory, arrayNumber)
			os.chdir(arrayNumber)
			if "{:03d}".format(channelNumber) not in os.listdir('.'):
				os.mkdir("{:03d}".format(channelNumber))
			path = os.path.join(path, "{:03d}".format(channelNumber))
			os.chdir("{:03d}".format(channelNumber))
			self.save() 
		os.chdir(directory)
		return self
	
	#TODO: Step size to split into different sub-figures; use next and previous to move through the time. 
	def plot(self, i = None, ax = None, overlay = False):
		self.current_idx = i 
		if ax is None: 
			ax = plt.gca()
		if not overlay:
			ax.clear()
		self.plotopts = {'LabelsOff': False, 'FFT': False, 'XLims': [0, 150]}
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
		direct = self.dirs[0]
		session = DPT.levels.get_shortname("session", direct)
		array = DPT.levels.get_shortname("array", direct)
		channel = DPT.levels.get_shortname("channel", direct)
		title = session + array + channel
		ax.set_title(title)
		return ax 