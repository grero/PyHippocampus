import numpy as np 
from scipy import signal 
import DataProcessingTools as DPT 
from . import rplraw
from .helperfunctions import computeFFT
from matplotlib.pyplot import gca
import os 

def lowPassFilter(analogData, samplingRate = 30000, resampleRate = 1000, lowFreq = 1, highFreq = 150, LFPOrder = 4):
    analogData = analogData.flatten()
    lfpsData = signal.resample_poly(analogData, resampleRate, samplingRate)
    fn = resampleRate / 2
    lowFreq = lowFreq / fn 
    highFreq = highFreq / fn 
    [b, a] = signal.butter(LFPOrder, [lowFreq, highFreq], 'bandpass')
    print("Applying low-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
    lfps = signal.filtfilt(b, a, lfpsData, padtype = 'odd', padlen = 3*(max(len(b),len(a))-1))
    return lfps, resampleRate

class RPLLFP(DPT.DPObject):

    filename = "rpllfp.hkl"
    argsList = [('ResampleRate', 1000), ('LFPOrder', 8), ('LowPassFrequency', [1, 150])]
    level = 'channel'

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        if type(self.args['LowPassFrequency']) == str:
            self.args['LowPassFrequency'] = list(map(int, self.args['LowPassFrequency'].split(",")))
        self.data = []
        self.analogInfo = {}
        self.numSets = 0 
        rw = rplraw.RPLRaw()
        if len(rw.data) > 0:
            # create object
            DPT.DPObject.create(self, *args, **kwargs)
            lfpData, resampleRate = lowPassFilter(rw.data, samplingRate = rw.analogInfo['SampleRate'], resampleRate = self.args['ResampleRate'], LFPOrder = int(self.args['LFPOrder'] / 2), lowFreq = self.args['LowPassFrequency'][0], highFreq = self.args['LowPassFrequency'][1])
            self.analogInfo['SampleRate'] = resampleRate
            self.analogInfo['MinVal'] = np.amin(lfpData)
            self.analogInfo['MaxVal'] = np.amax(lfpData)
            self.analogInfo['HighFreqCorner'] = self.args['LowPassFrequency'][0] * resampleRate
            self.analogInfo['LowFreqCorner'] = self.args['LowPassFrequency'][1] * resampleRate
            self.analogInfo['NumberSamples'] = len(lfpData)
            self.analogInfo['HighFreqOrder'] = self.args['LFPOrder']
            self.analogInfo['LowFreqOrder'] = self.args['LFPOrder']
            self.analogInfo['ProbeInfo'] = rw.analogInfo['ProbeInfo'].replace('raw', 'lfp')
            self.data = lfpData.astype('float32')
            self.numSets = 1 
        else:
            # create empty object
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)            
        return self 
        
    def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):

        plotOpts = {'LabelsOff': False, 'FFT': False, 'XLims': [0, 150], 'TimeSplit': 10, 'PlotAllData': False}

        for (k, v) in plotOpts.items():
            plotOpts[k] = kwargs.get(k, v)

        if getPlotOpts:
            return plotOpts

        if getNumEvents:
            # Return the number of events avilable
            if plotOpts['FFT'] or plotOpts['PlotAllData']:
                return 1, 0 
            else:
                if i is not None:
                    idx = i 
                else:
                    idx = 0 
                totalEvents = len(self.data) / (self.analogInfo['SampleRate'] * plotOpts['TimeSplit'])
                return totalEvents, i

        if getLevels:        
            # Return the possible levels for this object
            return ["channel", 'trial']

        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()

        self.analogTime = [(i * 1000) / self.analogInfo["SampleRate"] for i in range(len(self.data))]
    
        plot_type_FFT = plotOpts['FFT']
        if plot_type_FFT: 
            fftProcessed, f = plotFFT(self.data, self.analogInfo['SampleRate'])
            ax.plot(f, fftProcessed)
            if not plotOpts['LabelsOff']:
                ax.set_xlabel('Freq (Hz)')
                ax.set_ylabel('Magnitude')
            ax.set_xlim(plotOpts['XLims'])
        else:
            if plotOpts['PlotAllData']:
                ax.plot(self.analogTime, self.data)
            else: 
                idx = [self.analogInfo['SampleRate'] * plotOpts['TimeSplit'] * i, self.analogInfo['SampleRate'] * plotOpts['TimeSplit'] * (i + 1) + 1] 
                data = self.data[int(idx[0]):int(idx[1])]
                time = self.analogTime[int(idx[0]):int(idx[1])] 
                ax.plot(time, data)
            if not plotOpts['LabelsOff']:
                ax.set_ylabel('Voltage (uV)')
                ax.set_xlabel('Time (ms)')
        direct = os.getcwd()
        day = DPT.levels.get_shortname('day', direct)
        session = DPT.levels.get_shortname("session", direct)
        array = DPT.levels.get_shortname("array", direct)
        channel = DPT.levels.get_shortname("channel", direct)
        title = 'D' + day + session + array + channel
        ax.set_title(title)
        return ax 
