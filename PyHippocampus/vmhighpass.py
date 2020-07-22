import DataProcessingTools as DPT 
import numpy as np 
from .rplparallel import RPLParallel 
from .rplhighpass import RPLHighPass 

class VMHighPass(DPT.DPObject):

	filename = 'vmhighpass.hkl'
	argsList = []
	level = 'channel'

	def __init__(self, *args, **kwargs):
		DPT.DPObject.__init__(self, *args, **kwargs)

	def create(self, *args, **kwargs):
		self.data = []
		self.markers = []
		self.trialIndices = []
		self.timeStamps = []
		self.numSets = 0
		rh = RPLHighPass()
		rp = RPLParallel()
		if len(rh.data) > 0 and len(rp.timeStamps) > 0: 
			self.data = rh.data 
			self.markers = rp.markers
			self.timeStamps = rp.timeStamps
			self.trialIndices = rp.trialIndices
			self.numSets = rp.trialIndices.shape[0]
		return self 

	def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):
		pass 
