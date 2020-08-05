from .rplraw import RPLRaw 
from .rplparallel import RPLParallel
import DataProcessingTools as DPT 
import numpy as np 
from .helperfunctions import plotFFT
from .helperfunctions import removeLineNoise
import os 
import matplotlib.pyplot as plt 

class VMRaw(DPT.DPObject):
    
    filename = 'vmraw.hkl'
    level = 'channel'
    argsList = []

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        self.data = []
        self.markers = []
        self.trialIndices = []
        self.timeStamps = []
        self.numSets = 0
        rp = RPLParallel()
        if len(rp.timeStamps) > 0: 
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

        plotOpts = {'LabelsOff': False, 'PreTrial': 500, 'NormalizeTrial': False, 'RewardMarker': 3, 'TimeOutMarker': 4, 'PlotAllData': False, 'TitleOff': False, 'FreqLims': [], 'RemoveLineNoise': False, 'RemoveLineNoiseFreq': 10, 'LogPlot': False, "Type": DPT.objects.ExclusiveOptions(["FreqPlot", 'Signal'], 1)} 

        plot_type = plotOpts['Type'].selected()

        if getPlotOpts:
            return plotOpts 

        if getLevels:
            return ['trial', 'all']

        if getNumEvents:
            if plotOpts['PlotAllData']: # to avoid replotting the same data. 
                return 1, 0 
            if plot_type == 'FreqPlot' or plot_type == 'Signal' or plot_type == 'TFfft':
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
            ax.axvline((self.timeStamps[i][2] - self.timeStamps[i][0]) * 1000, color = 'r')

        elif plot_type == 'FreqPlot':
            if plotOpts['PlotAllData']:
                data = self.data 
            else: 
                data = self.data[idx[0]-1:idx[-1]]
            if plotOpts['RemoveLineNoise']:
                data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
            datam = np.mean(data)
            fftProcessed, f = plotFFT(data - datam, sRate)
            ax.plot(f[1:], fftProcessed[1:])
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
