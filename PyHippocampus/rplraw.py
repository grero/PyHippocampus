import neo  
from neo.io import BlackrockIO 
import numpy as np 
import os 
import DataProcessingTools as DPT 
# from rpllfp import RPLLFP 
# from rplhighpass import rplhighpass

# RPLSplit to give the data to rplraw. 
# If no data, rplraw to call rplsplit to open and pass it the data. 

class RPLRaw(DPT.DPObject):

	filename = 'rplraw.hkl'
	argsList = [('Data', True), ('analogInfo', {}), ('analogData', []), ('sessionEye', False)]

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self,normpath = False,  *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = []
		self.analogInfo = {}

		self.data = self.args['analogData']
		self.analogInfo = self.args['analogInfo']
		arrayNumber = self.analogInfo['ArrayNumber']
		channelNumber = self.analogInfo['ChannelNumber']
		self.numSets = 1

		if kwargs.get('saveLevel', 0) > 0:
			print('rplraw saving')
			arrayDir = "array{:02d}".format(int(arrayNumber))
			channelDir = "channel{:03d}".format(int(channelNumber))
			print(arrayDir)
			print(channelDir)
			directory = os.getcwd() # Go to the channel directory and save file there. 
			if arrayDir not in os.listdir('.'): 
				os.mkdir(arrayDir)
			#path = os.path.join(directory, arrayNumber)
			os.chdir(arrayDir)
			if channelDir not in os.listdir('.'):
				os.mkdir(channelDir)
			#path = os.path.join(path, channelDir)
			os.chdir(channelDir)
			self.save() 
			# if self.args['sessionEye']:
			# 	print('running rpllfp...')
			# 	RPLLFP(saveLevel = 1)
			# 	print('running rplhighpass...')
			# 	RPLHighPass(saveLevel = 1)
			os.chdir(directory)
		print(os.getcwd())
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
