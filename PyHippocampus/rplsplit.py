import numpy as np 
import neo  
from neo.io import BlackrockIO 
import os 
import glob 
from .rplraw import RPLRaw 
from .rpllfp import RPLLFP
from .rplparallel import RPLParallel
from .rplhighpass import RPLHighPass
import DataProcessingTools as DPT 
# from export_mountain_cells import export_mountain_cells
# from mountain_batch import mountain_batch

# Slurm 
# TODO: For EYE Submit a job takes rplraw creates 1) rplhighpass -> spike sorting and 2) rpllfp -> vmlfp -> time frequency analysis (one job each). 
# Use the eye session data to normalize the rpllfp from the navigation session.

class RPLSplit(DPT.DPObject):

	filename = 'rplsplit.hkl'
	argsList = [('channel', []), ('SkipLFP', False), ('SkipParallel', False), ('SkipHighPass', False), ('SessionEye', False), ('SkipSort', False), ('SkipHPC', False)] # Channel [] represents all channels to be processed, otherwise a list of channels to be provided.  
	level = 'session'

	def __init__(self, *args, **kwargs):
		rr = DPT.levels.resolve_level(self.level, os.getcwd())
		with DPT.misc.CWD(rr):
			DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):

		if not self.args['SkipParallel']: 
			print('Calling RPLParallel...')
			rp = RPLParallel(saveLevel = 1)

		ns5File = glob.glob('*.ns5')
		if len(ns5File) > 1: 
			print('Too many .ns5 files, do not know which one to use.')
			return 
		if len(ns5File) == 0:
			print('.ns5 file missing')
			return 
		reader = BlackrockIO(ns5File[0])
		# print('file opened') 
		bl = reader.read_block(lazy = True)
		segment = bl.segments[0]
		chx = bl.channel_indexes[2] # For the raw data.
		analogInfo = {} 
		analogInfo['SampleRate'] = float(segment.analogsignals[2].sampling_rate)
		annotations = chx.annotations
		names = list(map(lambda x: str(x), chx.channel_names))
		indexes = list(chx.index)

		def process_channel(data, annotations, chxIdx, analogInfo, channelNumber):
			analogInfo['ProbeInfo'] = names[chxIdx]
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
			analogInfo['ProbeInfo'] = names[chxIdx]
			arrayNumber = annotations['connector_ID'][chxIndex] + 1
			arrayDir = "array{:02d}".format(int(arrayNumber))
			channelDir = "channel{:03d}".format(int(channelNumber))
			directory = os.getcwd() # Go to the channel directory and save file there. 
			if arrayDir not in os.listdir('.'): 
				os.mkdir(arrayDir)
			os.chdir(arrayDir)
			if channelDir not in os.listdir('.'):
				os.mkdir(channelDir)
			os.chdir(channelDir)
			print('Calling RPLRaw for channel {:03d}'.format(channelNumber))
			RPLRaw(analogData = data, analogInfo = analogInfo, saveLevel = 1)
			if self.args['SessionEye']:
				if not self.args['SkipLFP']:
					print('Calling RPLLFP for channel {:03d}'.format(channelNumber))
					RPLLFP(saveLevel = 1)
				if not self.args['SkipHighPass']:
					print('Calling RPLHighPass for channel {:03d}'.format(channelNumber))
					RPLHighPass(saveLevel = 1)
			else:
				if not self.args['SkipLFP']:
					if self.args['SkipHPC']:
						print('Calling RPLLFP for {:03d}'.format(channelNumber))
						RPLLFP(saveLevel = 1)
					else:
						print('Adding slurm LFP script for channel {:03d} to job queue'.format(channelNumber))
						os.system('slurm-lfp.sh')
				if not self.args['SkipHighPass']:
					if not self.args['SkipSort']:
						if self.args['SkipHPC']:
							print('Calling RPLHighPass for channel {:03d}'.format(channelNumber))
							RPLHighPass(saveLevel = 1)
							print('Calling Mountain Sort for channel {:03d}'.format(channelNumber))
							# mountain_batch()
							# export_mountain_cells()
						else: 
							print('Adding slurm RPLHighPass script for channel {:03d} to job queue'.format(channelNumber))
							os.system('slurm-highpass.sh') 
					else: # Skipping sorting 
						if self.args['SkipHPC']:
							print('Calling RPLHighPass for {:03d}'.format(channelNumber))
							RPLHighPass(saveLevel = 1)
						else:
							print('Adding slurm RPLHighPass script for channel {:03d} to job queue'.format(channelNumber))
							os.system('slurm-highpass.sh') 
			os.chdir(directory)
			print('Channel {:03d} processed'.format(channelNumber))
			return 

		if len(self.args['channel']) == 0: 
			for chxIdx in chx: 
				data = segment.analogsignals[2].load(time_slice=None, channel_indexes=[chxIdx])
				number = list(filter(lambda x: chxIdx in x, names))[0]
				process_channel(data, annotations, chxIdx, analogInfo, number)
		else: 
			for i in self.args['channel']:
				chxIndex = names.index(list(filter(lambda x: str(i) in x, names))[0])
				data = np.array(segment.analogsignals[2].load(time_slice=None, channel_indexes=[chx.index[chxIndex]]))
				process_channel(data, annotations, chxIndex, analogInfo, i)
		return 

	def plot(self, i = None, ax = None, overlay = False):
		pass 

os.chdir('/Users/anujpatel/intern/PyHippocampus/PyHippocampus/181105/session01/')
rp = RPLSplit(saveLevel = 1, channel = [65, 85, 96], redoLevel = 1, SessionEye = False, SkipHPC = True)

