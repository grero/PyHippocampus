import numpy as np

class VMPlot:
    def __init__(self, *args, **kwargs):
        self.create(*args, **kwargs)
    
    def create(self, *args, **kwargs):
        # for user to input
        self.trial_idx = []
        self.ax = []
        self.plotOpts = []
        self.marker_multiplier = 1
        
        # protected field
        self._trial_start_time = []
        self._trial_cue_time = []
        self._trial_end_time = []
        self._data_timestamps = []
        
        # self.__dict__.update(kwargs)
        for (k,v) in kwargs.items():
            if hasattr(self, k):
                self.__dict__[k] = v
            else:
                raise ValueError("{0} does not exist vmplot...".format(k))
        
        self.get_timestamps()
        self._trial_start_time, self._trial_cue_time, self._trial_end_time = \
                self.timeStamps[self.trial_idx] * self.marker_multiplier        
        
    def get_timestamps(self):
        x = self.trialIndices
        if self.trial_idx == 0:
            self._data_timestamps = np.arange(0, x[self.trial_idx + 1][0].astype(int))
        elif self.trial_idx != x.shape[0] - 1:
            self._data_timestamps = np.arange(x[self.trial_idx - 1][2], x[self.trial_idx + 1][0]).astype(int)
        else:
            self._data_timestamps = np.arange(x[self.trial_idx - 1][2], len(self.data) - 1).astype(int)
    
    def plot_markers(self):
        self.ax.axvline(0, color='g') # Start of trial. 
        self.ax.axvline((self._trial_cue_time - self._trial_start_time), color='m')
        
        if np.floor(self.markers[self.trial_idx][2] / 10) == self.plotOpts['RewardMarker']:
            self.ax.axvline((self._trial_end_time - self._trial_start_time), color='b')
            
        elif np.floor(self.markers[self.trial_idx][2] / 10) == self.plotOpts['TimeOutMarker']:
            self.ax.axvline((self._trial_end_time - self._trial_start_time), color='r')    
            
    def get_data_timestamps_plot(self):
        return self._data_timestamps/self.samplingRate - self._trial_start_time