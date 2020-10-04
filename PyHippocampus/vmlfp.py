import numpy as np 
import DataProcessingTools as DPT 
from .rplparallel import RPLParallel
from .rpllfp import RPLLFP
from .helperfunctions import computeFFT, removeLineNoise
from .vmplot import VMPlot
import os 
import matplotlib.pyplot as plt 

class VMLFP(DPT.DPObject, VMPlot):

    filename = 'vmflp.hkl'
    argsList = []
    level = 'channel'

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)
        VMPlot.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        self.markers = np.array([])
        self.timeStamps = np.array([])
        self.trialIndices = np.array([])
        self.data = np.array([])
        self.samplingRate = None 
        self.numSets = 0  
        rp = RPLParallel()
        rlfp = RPLLFP()
        if len(rlfp.data) > 0 and len(rp.trialIndices) > 0: 
            # create object
            DPT.DPObject.create(self, *args, **kwargs)
            self.markers = rp.markers 
            self.samplingRate = rlfp.analogInfo['SampleRate']
            dsample = rp.samplingRate / rlfp.analogInfo['SampleRate']
            self.timeStamps = rp.timeStamps / dsample 
            self.trialIndices = np.rint(rp.trialIndices / dsample).astype('int')
            self.numSets = self.trialIndices.shape[0]
        else:
            # create empty object
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)            
        return self 

    def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):

        plotOpts = {'LabelsOff': False, 'NormalizeTrial': False, 'RewardMarker': 3,\
                    'TimeOutMarker': 4, 'PlotAllData': False, 'TitleOff': False,\
                    'FreqLims': [], 'RemoveLineNoise': False, 'RemoveLineNoiseFreq': 50,\
                    'LogPlot': False, 'TFfftWindow': 256, 'TFfftOverlap': 150,\
                    'TFfftPoints': 256, 'TFfftStart': 500, 'TFfftFreq': 150,\
                    "Type": DPT.objects.ExclusiveOptions(["FreqPlot", 'Signal', 'TFfft'], 1)} 

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
            
        sRate = self.samplingRate
        VMPlot.create(self, trial_idx=i, ax=ax, plotOpts=plotOpts, marker_multiplier=30)

        if i == None or i == 0:
            rlfp = RPLLFP()
            self.data = rlfp.data

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

        elif plot_type == 'TFfft': 
            if plotOpts['PlotAllData']:
                dIdx = self.trialIndices[:, -1] - self.trialIndices[:, 0]
                mIdx = np.amax(dIdx)
                spTimeStep = plotOpts['TFfftWindow'] - plotOpts['TFfftOverlap']
                spTimeBins = int(round(np.floor(mIdx/spTimeStep) - plotOpts['TFfftOverlap']/spTimeStep))
                nFreqs = (plotOpts['TFfftPoints'] / 2) + 1
                ops = np.zeros((int(nFreqs), spTimeBins))
                opsCount = np.zeros((int(nFreqs), spTimeBins)) 
                for j in range(self.numSets):
                    tftIdx = self.trialIndices[j,:]
                    data = self.data[int(tftIdx[0])-1:int(tftIdx[-1])]
                    if plotOpts['RemoveLineNoise']:
                        data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
                    datam = np.mean(data)
                    window = np.hamming(plotOpts['TFfftWindow'])
                    [s, f, t, im] = plt.specgram(data - datam, window = window, NFFT = plotOpts['TFfftPoints'], noverlap = plotOpts['TFfftOverlap'], Fs = sRate)
                    psIdx = range(0, s.shape[1])
                    ops[:, psIdx] = ops[:, psIdx] + s 
                    opsCount[:, psIdx] = opsCount[:, psIdx] + 1 
                x = np.arange(0, mIdx-1, plotOpts['TFfftWindow'] - plotOpts['TFfftOverlap'])
                x = x[:len(x) - 2]
                y = np.arange(0, (sRate / 2)+1, sRate / plotOpts['TFfftPoints'])
                i = ops / opsCount 
                im = ax.pcolormesh(x, y, i)
                ax.set_ylim([0, plotOpts['TFfftFreq']])
                # Uncomment colorbar line after PanGUI is fixed. 
                # plt.colorbar(im, ax = ax) 
            else: 
                tIdx = self.trialIndices[i,:]
                idx = [tIdx[0] - ((plotOpts['TFfftStart']+500)/1000*sRate), tIdx[0] - ((plotOpts['TFfftStart']+1)/1000*sRate)]
                data = self.data[int(idx[0])-1:int(idx[-1])]
                datam = np.mean(data)
                window = np.hamming(plotOpts['TFfftWindow'])
                [s, f, t, im] = plt.specgram(data - datam, window = window, NFFT = plotOpts['TFfftPoints'], noverlap = plotOpts['TFfftOverlap'], Fs = sRate)
                Pmean = np.mean(s, axis = 1)
                Pstd = np.std(s, axis = 1, ddof = 1)
                idx = [(tIdx[0] - (plotOpts['TFfftStart']/1000 * sRate)), tIdx[1], tIdx[2]]
                data = self.data[int(idx[0])-1:int(idx[-1])]
                datam = np.mean(data)
                window = np.hamming(plotOpts['TFfftWindow'])
                [s, f, t, im] = plt.specgram(data - datam, window = window, NFFT = plotOpts['TFfftPoints'], noverlap = plotOpts['TFfftOverlap'], Fs = sRate)
                spec_Pnorm = np.zeros(s.shape)
                for row in range(s.shape[0]):
                    spec_Pnorm[row, :] = (s[row, :] - Pmean[row]) / Pstd[row]
                spec_T = np.arange((-plotOpts['TFfftStart']/1000), t[-1] - (plotOpts['TFfftStart']/1000 + plotOpts['TFfftWindow']/sRate/2)+(plotOpts['TFfftWindow'] - plotOpts['TFfftOverlap'])/sRate, (plotOpts['TFfftWindow'] - plotOpts['TFfftOverlap'])/sRate)
                ax.axvline(0, color = 'k') 
                ax.axvline((self.timeStamps[i][1] - self.timeStamps[i][0]) * 30000 / 1000,  color = 'k')
                im = ax.pcolormesh(spec_T, f, spec_Pnorm, vmin = -10, vmax = 10)
                ax.set_ylim([0, plotOpts['TFfftFreq']])
                # Uncomment colour bar line after PanGUI is fixed             
                # plt.colorbar(im, ax = ax)

        if not plotOpts['LabelsOff']:
            if plot_type == 'FreqPlot':
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Magnitude')
            elif plot_type == 'TFfft':
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
            else:
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Voltage (uV)')

        if not plotOpts['TitleOff']:
            channel = DPT.levels.get_shortname("channel", os.getcwd())[1:]
            ax.set_title('channel' + str(channel))

        if len(plotOpts['FreqLims']) > 0:
            if plot_type == 'FreqPlot':
                ax.xlim(plotOpts['FreqLims'])
            elif plot_type == 'TFfft':
                ax.ylim(plotOpts['FreqLims'])
        return ax 
