import numpy as np 
import neo  
from neo.io import BlackrockIO 
import os 
import glob 
from . import RPLRaw
from . import RPLLFP 
from . import RPLParallel
from . import RPLHighPass
import DataProcessingTools as DPT 

# Slurm 
# TODO: For EYE Submit a job takes rplraw creates 1) rplhighpass -> spike sorting and 2) rpllfp -> vmlfp -> time frequency analysis (one job each). 
# Use the eye session data to normalize the rpllfp from the navigation session.

class RPLSplit(DPT.DPObject):

	filename = 'rplsplit.hkl'
	argsList = [('channel', []), ('SkipLFP', False), ('SkipParallel', False), ('SkipHighPass', False), ('sessionEye', False), ('SkipSort', False), ('SkipHPC', False)] # Channel [] represents all channels to be processed, otherwise a list of channels to be provided.  
	level = 'session'

	def __init__(self, *args, **kwargs):
		rr = DPT.levels.resolve_level("session", os.getcwd())
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
				data = segment.analogsignals[2].load(time_slice=None, channel_indexes=[chxIdx])
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
				analogInfo['ProbeInfo'] = names[chxIdx]
				analogInfo['ChannelNumber'] = list(filter(lambda x: chxIdx in x, names))[0]
				RPLRaw(analogdata = data, analoginfo = analogInfo)
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
				arrayNumber = annotations['connector_ID'][chxIndex] + 1
				channelNumber = i 
				analogInfo['ProbeInfo'] = names[chxIndex]
				print(analogInfo)
				print('Calling RPLRaw')
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
				RPLRaw(analogdata = data, analoginfo = analogInfo, saveLevel = 1)
			if not self.args['SkipHPC']:
				if self.args['sessionEye']:
					if not self.args['SkipLFP']:
						print('calling lfp') 
						RPLLFP(saveLevel = 1)
					if not self.args['SkipHighPass']:
						print('calling highpass')
						RPLHighPass(saveLevel = 1) 
				if not self.args['sessionEye']:
					if not self.args['SkipLFP']: 
						os.system("sbatch slurm-lfp.sh")
					if not self.args['SkipHighPass']:
						os.system("sbatch slurm-highpass.sh") 
			os.chdir(directory)
			print(os.getcwd())
		return 

	def plot(self, i = None, ax = None, overlay = False):
		pass 


