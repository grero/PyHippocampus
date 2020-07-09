import numpy as np 
import neo  
from neo.io import BlackrockIO 
import os 
import glob 
from rplraw import RPLRaw
# from rpllfp import RPLLFP 
from rplparallel import RPLParallel
#from rplhighpass import RPLHighPass
import DataProcessingTools as DPT 

# RPLSplit to give the data to rplraw. 
# If no data, rplraw to call rplsplit to open and pass it the data. 

# Slurm 
# TODO: For EYE Submit a job takes rplraw creates 1) rplhighpass -> spike sorting and 2) rpllfp -> vmlfp -> time frequency analysis (one job each). 
# Use the eye session data to normalize the rpllfp from the navigation session.

class RPLSplit(DPT.DPObject):

	filename = 'rplsplit.hkl'
	argsList = [('channel', []), ('SkipLFP', False), ('SkipParallel', False), ('SkipHighPass', False), ('sessionEye', False)] # Channel [] represents all channels to be processed, otherwise a list of channels to be provided.  

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, normpath = False, *args, **kwargs)

	def create(self, *args, **kwargs):

		# To check whether it is being called from the session or the channel directory. 
		cwd = os.getcwd()
		level = DPT.levels.level(cwd)
		if level == 'channel':
			rr = DPT.levels.resolve_level('session', cwd)
			# channelNumber = int(DPT.levels.get_level_name('channel', cwd)[-3:])
			os.chdir(rr) # Move to the session directory. 

		ns5File = glob.glob('*.ns5')
		if len(ns5File) > 1: 
			print('Too many .ns5 files, do not know which one to use.')
			return 
		if len(ns5File) == 0:
			print('.ns5 file missing')
			return 
		reader = BlackrockIO(ns5File[0])
		print('file opened') 
		bl = reader.read_block(lazy = True)
		segment = bl.segments[0]
		chx = bl.channel_indexes[2] # For the raw data.
		analogInfo = {} 
		analogInfo['SampleRate'] = float(segment.analogsignals[2].sampling_rate)
		annotations = chx.annotations
		names = list(map(lambda x: str(x), chx.channel_names))
		indexes = list(chx.index)
		if len(self.args['channel']) == 0: 
			for chxIdx in chx: 
				data = segment.analogsignals[2].load(time_slice=None, channel_indexes=[chx.index[chxIndex]])
				analogInfo['Units'] = 'uV'
				analogInfo['HighFreqCorner'] = annotations['nev_hi_freq_corner'][chxIndex]
				analogInfo['HighFreqOrder'] = annotations['nev_hi_freq_order'][chxIndex]
				analogInfo['HighFilterType'] = annotations['nev_hi_freq_type'][chxIndex]
				analogInfo['LowFreqCorner'] = annotations['nev_lo_freq_corner'][chxIndex]
				analogInfo['LowFreqOrder'] = annotations['nev_lo_freq_order'][chxIndex]
				analogInfo['LowFilterType'] = annotations['nev_lo_freq_type'][chxIndex]
				analogInfo['MaxVal'] = np.amax(data)
				analogInfo['MinVal'] = np.amin(data)
				analogInfo['NumberSamples'] = len(data)
				analogInfo['ArrayNumber'] = annotations['connector_ID'][chxIndex] + 1
				analogInfo['ProbeName'] = names[chxIdx]
				analogInfo['ChannelNumber'] = list(filter(lambda x: chxIdx in x, names))[0]
				RPLRaw(Data = True, analogData = data, analogInfo = analogInfo)
				# Add shell command to add RPLLFP and RPLHighpass to the queue. 
		else: 
			for i in self.args['channel']:
				chxIndex = names.index(list(filter(lambda x: str(i) in x, names))[0])
				data = np.array(segment.analogsignals[2].load(time_slice=None, channel_indexes=[chx.index[chxIndex]]))
				analogInfo['Units'] = 'uV'
				analogInfo['HighFreqCorner'] = annotations['nev_hi_freq_corner'][chxIndex]
				analogInfo['HighFreqOrder'] = annotations['nev_hi_freq_order'][chxIndex]
				analogInfo['HighFilterType'] = annotations['nev_hi_freq_type'][chxIndex]
				analogInfo['LowFreqCorner'] = annotations['nev_lo_freq_corner'][chxIndex]
				analogInfo['LowFreqOrder'] = annotations['nev_lo_freq_order'][chxIndex]
				analogInfo['LowFilterType'] = annotations['nev_lo_freq_type'][chxIndex]
				analogInfo['MaxVal'] = np.amax(data)
				analogInfo['MinVal'] = np.amin(data)
				analogInfo['NumberSamples'] = len(data)
				analogInfo['ArrayNumber'] = annotations['connector_ID'][chxIndex] + 1
				analogInfo['ChannelNumber'] = i 
				analogInfo['ProbeName'] = names[chxIndex]
				print(analogInfo)
				print('calling rplraw')
				RPLRaw(Data = True, analogData = data, analogInfo = analogInfo, saveLevel = 1)
				# Add shell command to add RPLLFP and RPLHighpass to the queue. 
				# if not self.args['SkipLFP']:
				# 	rlfp = RPLLFP()
				# if not self.args['SkipHighPass']:
				# 	rhp = RPLHighpass()
		print('calling rplparallel')
		if not self.args['SkipParallel']: 
			rp = RPLParallel(saveLevels = 1)
		return  

	def plot(self, i = None, ax = None, overlay = False):
		pass 

rl = RPLSplit(saveLevel = 1, channel = [10, 2, 9])
