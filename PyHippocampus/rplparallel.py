import neo  
from neo.io import BlackrockIO 
import numpy as np 
import glob 
import DataProcessingTools as DPT 
import PanGUI

# Call constructor with data to save. 
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
	argsList = [('ObjectLevel', 'Session')]

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, normpath = False, *args, **kwargs) 

	def create(self, *args, **kwargs):
		self.markers = []
		self.rawMarkers = []
		self.timeStamps = []
		self.trialIndices = []
		self.sessionStartTime = None 
		self.samplingRate = None 
		self.numSets = 0 

		nevFile = glob.glob("*.nev")
		if len(nevFile) == 0:
			print("No .nev files in directory. Returning empty object..")
			if kwargs.get("saveLevel", 0) > 0:
				self.save()
			return self 
		else: 
			reader = BlackrockIO(nevFile[0])
			ev_rawtimes, _, ev_markers = reader.get_event_timestamps()
			ev_times = reader.rescale_event_timestamp(ev_rawtimes, dtype = "float64")
			self.rawMarkers = ev_markers
			self.sessionStartTime = ev_times[0]
			self.numSets = 1
			try: 
				self.markers, self.timeStamps, self.trialIndices = arrangeMarkers(ev_markers, ev_times)
			except: 
				print('problem with arrange markers.')
			if kwargs.get("saveLevel", 0) > 0:
				self.save()
			return self 

	def plot(self, i = None, ax = None, overlay = False):
		self.current_idx = i 
		if ax is None: 
			ax = plt.gca()
		if not overlay:
			ax.clear()
		self.plotopts = {"labelsOff": False, "groupPlots": 1, "groupPlotIndex": 1, "color":'b', 'M1':20, "M2":30}
		raw_markers = self.markers.flatten()
		markers = list(raw_markers[::3])
		markers.extend(list(raw_markers[1::3])) 
		markers.extend(list(raw_markers[2::3]))
		ax.stem(markers)
		ax.hlines(self.plotopts['M1'], 0, len(markers), color = 'blue')
		ax.hlines(self.plotopts['M2'], 0, len(markers), color = 'red')
		if not self.plotopts['labelsOff']: 
			ax.set_xlabel("Marker Number")
			ax.set_ylabel('Marker Value')
		return ax 
