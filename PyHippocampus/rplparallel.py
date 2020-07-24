import neo  
from neo.io import BlackrockIO 
import numpy as np 
import glob 
import DataProcessingTools as DPT 
import os

def arrangeMarkers(markers, timeStamps, samplingRate = 30000):
	rawMarkers = markers 
	rm1 = np.reshape(rawMarkers, (2, -1), order = "F") # Reshape into two rows. 
	if rm1[1, 0] == 0:
		if rm1[0, 0] > 1: 
			if rm1[0, 0] == 204 or rm1[0, 0] == 84: # Format 204 or 84, MATLAB code is identical for both cases. 
				markers = np.transpose(np.reshape(rm1[0,1:], (3, -1), order = "F"))
				rtime = timeStamps[2:]
				rt1 = np.reshape(rtime, (6, -1), order = "F")
				timeStamps = np.transpose(rt1[np.array([0, 2, 4]), :])
				trialIndices = np.floor(timeStamps * samplingRate).astype('int64')
				return markers, timeStamps, trialIndices
			else: 
				markers = np.transpose(np.reshape(rm1[0, :], (3, -1), order = "F"))
				rtime = timeStamps
				rt1 = np.reshape(rtime, (6, -1), order = "F")
				timeStamps = np.transpose(rt1[np.array([0, 2, 4]), :])
				trialIndices = np.floor(timeStamps * samplingRate).astype('int64')
				return markers, timeStamps, trialIndices
		else: 
			if rm1[0, 1] == 2:
				markers = np.transpose(np.reshape(rm1[0, :], (3, -1), order = "F"))
				rtime = timeStamps
				rt1 = np.reshape(rtime, (6, -1), order = "F")
				timeStamps = np.transpose(rt1[np.array([0, 2, 4]), :])
				trialIndices = np.floor(timeStamps * samplingRate).astype('int64')
				return markers, timeStamps, trialIndices
			else:
				markers = np.transpose(np.reshape(rm1[0, :], (2, -1), order = "F"))
				rtime = timeStamps
				rt1 = np.reshape(rtime, (2, -1))
				timeStamps = np.transpose(np.reshape(rt1[0, :], (2, -1), order = "F"))
				trialIndices = np.floor(timeStamps * samplingRate).astype('int64')
				return markers, timeStamps, trialIndices
	else: 
		markers = np.transpose(rm1)
		timeStamps = np.reshape(timeStamps, (2, -1), order = "F")
		trialIndices = np.floor(np.transpose(np.reshape(timeStamps * samplingRate, (2, -1), order = "F"))).astype('int64')
		return markers, timeStamps, trialIndices

class RPLParallel(DPT.DPObject):

	filename = 'rplparallel.hkl'
	argsList = []
	level = 'session'

	def __init__(self, *args, **kwargs):
		rr = DPT.levels.resolve_level(self.level, os.getcwd())
		with DPT.misc.CWD(rr):
			DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.markers = []
		self.timeStamps = []

		if 'data' in kwargs.keys():
			if kwargs['data']:
				self.markers = kwargs['markers'] 
				self.timeStamps = kwargs['timeStamps']
				self.trialIndices = kwargs['trialIndices']
				self.rawMarkers = kwargs['rawMarkers']
				self.sessionStartTime = kwargs['sessionStartTime']
				self.samplingRate = kwargs['sampleRate']
				self.numSets = 1
				return self 

		nevFile = glob.glob("*.nev")
		if len(nevFile) == 0:
			print("No .nev files in directory. Returning empty object...")
			self.rawMarkers = []
			self.trialIndices = []
			self.sessionStartTime = None 
			self.samplingRate = 30000 
			self.numSets = 0 
			return self 		
		else: 
			reader = BlackrockIO(nevFile[0])
			print('Opening .nev file, creating new RPLParallel object...')
			ev_rawtimes, _, ev_markers = reader.get_event_timestamps()
			ev_times = reader.rescale_event_timestamp(ev_rawtimes, dtype = "float64")
			if ev_markers[0] == 128:
				self.markers = ev_markers[::2]
				self.timeStamps = ev_times[::2]
				return self 
			self.samplingRate = 30000
			self.rawMarkers = ev_markers
			self.sessionStartTime = ev_times[0]
			self.numSets = 1
			try: 
				self.markers, self.timeStamps, self.trialIndices = arrangeMarkers(ev_markers, ev_times)
			except: 
				print('problem with arrange markers.')
			return self 

	def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):
		self.current_idx = i 

		plotopts = {"labelsOff": False, 'M1':20, "M2":30}

		for (k, v) in self.plotopts.items():
			plotopts[k] = kwargs.get(k, v)

		if getPlotOpts:
			return plotopts

		if getNumEvents:
			# Return the number of events avilable
			return 0, 1

		if getLevels:        
			# Return the possible levels for this object
			return ["session"]

		if ax is None: 
			ax = plt.gca()
		if not overlay:
			ax.clear()

		markers = self.rawMarkers[2::2]
		ax.stem(markers)
		ax.hlines(plotopts['M1'], 0, len(markers), color = 'blue')
		ax.hlines(plotopts['M2'], 0, len(markers), color = 'red')
		if not plotopts['labelsOff']: 
			ax.set_xlabel("Marker Number")
			ax.set_ylabel('Marker Value')
		return ax 
