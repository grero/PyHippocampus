from .rplraw import RPLRaw 
from .rplparallel import RPLParallel
import DataProcessingTools as DPT 
import numpy as np 
from .helperfunctions import computeFFT, removeLineNoise
from .vmplot import VMPlot
import os 
import matplotlib.pyplot as plt 

class VMRaw(DPT.DPObject, VMPlot):
    
    filename = 'vmraw.hkl'
    level = 'channel'
    argsList = []

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)
        VMPlot.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        self.data = []
        self.markers = []
        self.trialIndices = []
        self.timeStamps = []
        self.numSets = 0
        rp = RPLParallel()
        if len(rp.trialIndices) > 0: 
            # create object
            DPT.DPObject.create(self, *args, **kwargs)
            self.markers = rp.markers
            self.timeStamps = rp.timeStamps
            self.trialIndices = rp.trialIndices
            self.numSets = rp.trialIndices.shape[0]
        else:
            # create empty object
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)            
        return self 

    def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):
        plotOpts = {'LabelsOff': False, 'RewardMarker': 3, 'TimeOutMarker': 4,\
                    'PlotAllData': False, 'TitleOff': False, 'FreqLims': [],\
                    'RemoveLineNoise': False, 'RemoveLineNoiseFreq': 50, \
                    'LogPlot': False, "Type": DPT.objects.ExclusiveOptions(["FreqPlot", 'Signal'], 1)} 

        for (k, v) in plotOpts.items():
            plotOpts[k] = kwargs.get(k, v)

        plot_type = plotOpts['Type'].selected()

        if getPlotOpts:
            return plotOpts 

        if getLevels:
            return ['trial', 'all']

        if getNumEvents:
            if plotOpts['PlotAllData']: # to avoid replotting the same data. 
                return 1, 0 
            if plot_type == 'FreqPlot' or plot_type == 'Signal': 
                if i is not None:
                    nidx = i 
                else:
                    nidx = 0
                return self.numSets, nidx 

        if ax is None:
            ax = plt.gca()

        if not overlay:
            ax.clear()

        if i == None or i == 0:
            rw = RPLRaw()
            self.data = rw.data.flatten()
            self.samplingRate = rw.analogInfo['SampleRate']

        sRate = self.samplingRate
        VMPlot.create(self, trial_idx=i, ax=ax, plotOpts=plotOpts)

        if plot_type == 'Signal':
            data = self.data[self._data_timestamps]
            if plotOpts['RemoveLineNoise']:
                data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
            ax.plot(self.get_data_timestamps_plot(), data)
            self.plot_markers()
            

        elif plot_type == 'FreqPlot':
            if plotOpts['PlotAllData']:
                data = self.data 
            else: 
                data = self.data[self._data_timestamps]
            if plotOpts['RemoveLineNoise']:
                data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
            datam = np.mean(data)
            fftProcessed, f = computeFFT(data - datam, sRate)
            ax.plot(f, fftProcessed)
            if plotOpts['LogPlot']:
                ax.set_yscale('log')

        if not plotOpts['LabelsOff']:
            if plot_type == 'FreqPlot':
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
            else:
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Voltage (uV)')

        if not plotOpts['TitleOff']:
            channel = DPT.levels.get_shortname("channel", os.getcwd())[1:]
            ax.set_title('channel' + str(channel))

        if len(plotOpts['FreqLims']) > 0:
            if plot_type == 'FreqPlot':
                ax.xlim(plotOpts['FreqLims'])
        return ax
