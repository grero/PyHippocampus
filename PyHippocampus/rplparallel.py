import neo  
from neo.io import BlackrockIO 
import numpy as np 
import glob 
import DataProcessingTools as DPT 
import PanGUI

def arrangeMarkers(markers, timeStamps, samplingRate = 30000):
	rawMarkers = markers 
	rm1 = np.reshape(rawMarkers, (2, -1)) # Reshape into two rows. 
	print(np.reshape(rm1[0, 2:], (3, -1)))
	if rm1[1, 0] == 0:
		if rm1[0, 0] > 1: 
			if rm1[0, 0] == 204 or rm1[0, 0] == 84: # Format 204 or 84, MATLAB code is identical for both cases. 

				markers = np.transpose(np.reshape(rm1[0,2:], (3, -1)))
				print(markers)
				rtime = timeStamps[2:]
				rt1 = np.reshape(rtime, (6, -1))
				timeStamps = np.transpose(rt1[np.array([0, 2, 4]), :])
				trialIndices = np.floor(timeStamps * samplingRate)
				return markers, timeStamps, trialIndices
			else: 
				markers = np.transpose(np.reshape(rm1[0, :], (-1, 3)))
				rtime = timeStamps
				rt1 = np.reshape(rtime, (-1, 6))
				timeStamps = np.transpose(rt1[np.array([0, 2, 4]), :])
				trialIndices = np.floor(timeStamps * samplingRate)
				return markers, timeStamps, trialIndices
		else: 
			if rm1[0, 1] == 2:
				markers = np.transpose(np.reshape(rm1[0, :], (-1, 3)))
				rtime = timeStamps
				rt1 = np.reshape(rtime, (-1, 6))
				timeStamps = np.transpose(rt1[np.array([0, 2, 4]), :])
				trialIndices = np.floor(timeStamps * samplingRate)
				return markers, timeStamps, trialIndices
			else:
				markers = np.transpose(np.reshape(rm1[0, :], (-1, 2)))
				rtime = timeStamps
				rt1 = np.reshape(rtime, (-1, 2))
				timeStamps = np.transpose(np.reshape(rt1[0, :], (-1, 2)))
				trialIndices = np.floor(timeStamps * samplingRate)
				return markers, timeStamps, trialIndices
	else: 
		markers = np.transpose(rm1)
		timeStamps = np.reshape(timeStamps, (-1, 2))
		trialIndices = np.floor(np.transpose(np.reshape(timeStamps * samplingRate, (-1, 2))))
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

		nevFile = glob.glob("*.nev")
		if len(nevFile) > 1: 
			print("Too many .nev files, do not know which one to use.")
			return 
		if len(nevFile) == 0:
			print("No .nev files in directory.")
			return 
		reader = BlackrockIO(nevFile[0])
		ev_rawtimes, _, ev_markers = reader.get_event_timestamps()
		ev_times = reader.rescale_event_timestamp(ev_rawtimes, dtype = "float64")
		self.rawMarkers = ev_markers
		self.sessionStartTime = ev_times[0]
		try: 
			self.markers, self.timeStamps, self.trialIndices = arrangeMarkers(ev_markers, ev_times)
		except: 
			print('problem with arrange markers.')
		# self.timeStamps = timeStamps
		# self.markers = markers
		# self.trialIndices = trialIndices
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

# rp = RPLParallel(saveLevel = 1, redoLevel = 1)
# # print(rp.markers.shape)
# # print(rp.trialIndices.shape)
# # print(rp.timeStamps.shape)
# app = PanGUI.create_window(rp)