import DataProcessingTools as DPT 
import numpy as np 
from .rplparallel import RPLParallel 
from .spiketrain import Spiketrain
from .rplhighpass import RPLHighPass 
from .helperfunctions import plotFFT, removeLineNoise
import os 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

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

        plotOpts = {'LabelsOff': False, 'PreTrial': 500, 'RewardMarker': 3, 'TimeOutMarker': 4, 'PlotAllData': False, 'TitleOff': False, 'FreqLims': [], 'RemoveLineNoise': False, 'RemoveLineNoiseFreq': 10, 'LogPlot': True, 'SpikeTrain': True, "Type": DPT.objects.ExclusiveOptions(["FreqPlot", 'Signal'], 1)} 

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
        trialIndicesForN = self.trialIndices[i, :] 
        idx = [int(trialIndicesForN[0] - ((plotOpts['PreTrial'] / 1000) * sRate))] + list(trialIndicesForN[1:])

        if plot_type == 'Signal':
            data = self.data[idx[0]-1:idx[-1]]
            if plotOpts['RemoveLineNoise']:
                data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
            x = np.linspace(-plotOpts['PreTrial'], 0, num = int(plotOpts['PreTrial']*sRate/1000))
            x = np.concatenate((x, np.linspace(0, (len(data) - len(x))*1000/sRate, num = int(len(data) - len(x)))))
            ax.plot(x, data)
            ax.axvline(0, color = 'g') # Start of trial. 
            ax.axvline((self.timeStamps[i][1] - self.timeStamps[i][0]) * 1000, color = 'm')
            if np.floor(self.markers[i][2] / 10) == plotOpts['RewardMarker']:
                ax.axvline((self.timeStamps[i][2] - self.timeStamps[i][0]) * 1000, color = 'b')
            elif np.floor(self.markers[i][2] / 10) == plotOpts['TimeOutMarker']:
                ax.axvline((self.timeStamps[i][2] - self.timeStamps[i][0]) * 1000, color = 'r')
            if plotOpts['SpikeTrain']:
                st = DPT.objects.processDirs(None, Spiketrain)
                if st.numSets > 0: 
                    trialSpikes = [list(filter(lambda x: x >= (self.timeStamps[i][0] * 1000 - plotOpts['PreTrial']) and x <= self.timeStamps[i][2] * 1000, map(float, j))) for j in st.spiketimes] 
                    trialSpikes = [list(map(lambda x: round(x - self.timeStamps[i][0] * 1000, 3), k)) for k in trialSpikes]
                    colors = cm.rainbow(np.linspace(0, 1, len(trialSpikes)))
                    y_coords = [[int(np.amax(data)) + 5 * (k + 1) for j in range(len(trialSpikes[k]))] for k in range(len(trialSpikes))]
                    for trial, y, c in zip(trialSpikes, y_coords, colors): 
                        ax.scatter(trial, y, color = c, marker = '|') 

        elif plot_type == 'FreqPlot':
            if plotOpts['PlotAllData']:
                data = self.data 
            else: 
                data = self.data[idx[0]-1:idx[-1]]
            if plotOpts['RemoveLineNoise']:
                data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
            datam = np.mean(data)
            fftProcessed, f = plotFFT(data - datam, sRate)
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