import DataProcessingTools as DPT 
import numpy as np 
from .rplparallel import RPLParallel 
from .spiketrain import Spiketrain
from .rplhighpass import RPLHighPass 
from .helperfunctions import computeFFT, removeLineNoise
from .vmplot import VMPlot
import os 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

class VMHighPass(DPT.DPObject, VMPlot):

    filename = 'vmhighpass.hkl'
    argsList = []
    level = 'channel'

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

    def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, 
             getPlotOpts = False, overlay = False, **kwargs):

        plotOpts = {'LabelsOff': False, 'PreTrial': 500, 'RewardMarker': 3, 
                    'TimeOutMarker': 4, 'PlotAllData': False, 'TitleOff': False, 
                    'FreqLims': [], 'RemoveLineNoise': False, 
                    'RemoveLineNoiseFreq': 50, 'LogPlot': True, 
                    'SpikeTrain': False, 
                    "Type": DPT.objects.ExclusiveOptions(["FreqPlot", 'Signal'], 1)} 

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
            rh = RPLHighPass()
            self.data = rh.data
            self.samplingRate = rh.analogInfo['SampleRate']

        sRate = self.samplingRate
        VMPlot.create(self, trial_idx=i, ax=ax, plotOpts=plotOpts)

        if plot_type == 'Signal':
            data = self.data[self._data_timestamps]
            if plotOpts['RemoveLineNoise']:
                data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
            ax.plot(self.get_data_timestamps_plot(), data)
            self.plot_markers()
            if plotOpts['SpikeTrain']:
                st = DPT.objects.processDirs(None, Spiketrain)
                if st.numSets > 0: 
                    trialSpikes = [list(filter(lambda x: x >= (self._data_timestamps[0] * 1000 / sRate) and x <= (self._data_timestamps[-1] * 1000 / sRate), map(float, j))) for j in st.spiketimes] 
                    trialSpikes = [list(map(lambda x: round((x - self.timeStamps[i][0] * 1000) / 1000, 3), k)) for k in trialSpikes]
                    colors = cm.rainbow(np.linspace(0, 1, len(trialSpikes)))
                    y_coords = [[int(np.amax(data)) + 5 * (k + 1) for j in range(len(trialSpikes[k]))] for k in range(len(trialSpikes))]
                    for trial, y, c in zip(trialSpikes, y_coords, colors): 
                        ax.scatter(trial, y, color = c, marker = '|') 

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
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Voltage (uV)')

        if not plotOpts['TitleOff']:
            channel = DPT.levels.get_shortname("channel", os.getcwd())[1:]
            ax.set_title('channel' + str(channel))

        if len(plotOpts['FreqLims']) > 0:
            if plot_type == 'FreqPlot':
                ax.xlim(plotOpts['FreqLims'])
        return ax 