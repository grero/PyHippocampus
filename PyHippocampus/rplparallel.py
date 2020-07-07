import neo  
from neo.io import BlackrockIO 
import numpy as np 
import os 
import glob 
import h5py as h5 
import DataProcessingTools as DPT 

class RPLParallel(DPT.DPObject):
	def __init__(self):
		DPT.DPObject.__init__(self) 
		self.markers = []
		self.rawMarkers = []
		self.timeStamps = []
		self.trialIndices = []
		self.sessionStartTime = None 
		self.samplingRate = None 
		return 

	def plot(labelsOff = False, groupPlots = 1, groupPlotIndex = 1, colour = 'b', M1 = 20, M2 = 30): 
		# Function is incomplete. 
		self.plotopts['M1'] = M1 
		self.plotopts['M2'] = M2
		raw_markers = self.markers.flatten()
		markers = list(raw_markers[::3])
		markers.extend(list(raw_markers[1::3])) 
		markers.extend(list(raw_markers[2::3]))
		plt.stem(markers)
		plt.hlines(self.plotopts['M1'], 0, len(markers), color = 'blue')
		plt.hlines(self.plotopts['M2'], 0, len(markers), color = 'red')
		if not labelsOff: 
			plt.set_xlabel("Marker Number")
			plt.set_ylabel('Marker Value')
		return  

def arrangeMarkers(markers, timeStamps, samplingRate = 30000):
	rawMarkers = markers 
	rm1 = np.reshape(rawMarkers, (-1, 2)) # Reshape into two rows. 
	if rm1[1, 0] == 0:
		if rm1[0, 0] > 1: 
			if rm1[0, 0] == 204 or rm1[0, 0] == 84: # Format 204 or 84, MATLAB code is identical for both cases. 
				markers = np.transpose(np.reshape(rm1[0,2:], (-1, 3)))
				rtime = timeStamps[3:]
				rt1 = np.reshape(rtime, (-1, 6))
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

def rplparallel(saveLevel = 0, redoLevel = 0, samplingRate = 30000): 
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
	rawMarkers = ev_markers
	sessionStartTime = ev_times[0]
	try: 
		markers, timeStamps, trialIndices = arrangeMarkers(rawMarkers, ev_times)
	except: 
		print('problem with arrange markers.')
	rp = RPLParallel()
	rp.markers = markers 
	rp.rawMarkers = rawMarkers
	rp.timeStamps = timeStamps
	rp.trialIndices = trialIndices
	rp.samplingRate = samplingRate
	rp.sessionStartTime = sessionStartTime
	if saveLevel > 0:
		rp.save()
		print("rplparallel.hkl file saved!")
	return rp 
