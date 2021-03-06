from . import rplraw
from scipy import signal 
import numpy as np 
import DataProcessingTools as DPT 
from .helperfunctions import plotFFT
import matplotlib.pyplot as plt
import os 

def highPassFilter(analogData, samplingRate = 30000, lowFreq = 500, highFreq = 7500, HPOrder = 4):
    analogData = analogData.flatten()
    fn = samplingRate / 2
    lowFreq = lowFreq / fn 
    highFreq = highFreq / fn 
    [b, a] = signal.butter(HPOrder, [lowFreq, highFreq], 'bandpass')
    print("Applying high-pass filter with frequencies {} and {} Hz".format(lowFreq * fn, highFreq * fn))
    hps = signal.filtfilt(b, a, analogData, padtype = 'odd', padlen = 3*(max(len(b),len(a))-1))
    return hps, samplingRate

class RPLHighPass(DPT.DPObject):

    filename = "rplhighpass.hkl"
    argsList = [("HighOrder", 8), ("HighPassFrequency", [500, 7500])]
    level = "channel"

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        if type(self.args['HighPassFrequency']) == str:
            self.args['HighPassFrequency'] = list(map(int, self.args['HighPassFrequency'].split(",")))
        self.data = []
        self.analogInfo = {}
        self.numSets = 0
        rw = rplraw.RPLRaw()
        if len(rw.data) > 0: 
            # create object
            DPT.DPObject.create(self, *args, **kwargs)
            hpData, samplingRate = highPassFilter(rw.data, samplingRate = rw.analogInfo['SampleRate'], HPOrder = int(self.args['HighOrder'] / 2), lowFreq = self.args['HighPassFrequency'][0], highFreq = self.args['HighPassFrequency'][1])
            self.analogInfo['SampleRate'] = samplingRate
            self.analogInfo['MinVal'] = np.amin(hpData)
            self.analogInfo['MaxVal'] = np.amax(hpData)
            self.analogInfo['HighFreqCorner'] = self.args['HighPassFrequency'][0] * samplingRate
            self.analogInfo['LowFreqCorner'] = self.args['HighPassFrequency'][1] * samplingRate
            self.analogInfo['NumberSamples'] = len(hpData)
            self.analogInfo['HighFreqOrder'] = self.args['HighOrder']
            self.analogInfo['LowFreqOrder'] = self.args['HighOrder']
            self.analogInfo['ProbeInfo'] = rw.analogInfo['ProbeInfo'].replace('raw', 'hp')
            self.data = hpData.astype('float32')
            self.numSets = 1 
        else:
            # create empty object
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)            
        return self

    def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):

        if ax is None: 
            ax = plt.gca()
        if not overlay:
            ax.clear()

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

