from .rplraw import RPLRaw 
import DataProcessingTools as DPT 
from .helperfunctions import plotFFT
import os 
import numpy as np 

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
        if len(rh.data) > 0 and len(rp.timeStamps) > 0: 
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

        plotOpts = {'LabelsOff': False, 'PreTrial': 500, 'NormalizeTrial': False, 'RewardMarker': 3, 'TimeOutMarker': 4, 'PlotAllData': False, 'TitleOff': False, 'FreqLims': [], 'RemoveLineNoise': False, 'RemoveLineNoiseFreq': 10, 'LogPlot': False, 'TFfttWindow': 200, 'TFfttOverlap': 150, 'TFfftPoints': 256, 'TFfftStart': 500, 'TFfftFreq': 150, "Type": DPT.objects.ExclusiveOptions(["FreqPlot", 'Signal', 'TFfft'], 1)} 
        # Add flag for removelinenoise and a specific value. 

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
        
        # Magintude of the complex number and square it -> power density. 

        sRate = self.samplingRate
        trialIndicesForN = self.trialIndices[i, :] 
        idx = [int(trialIndicesForN[0] - ((plotOpts['PreTrial'] / 1000) * sRate))] + list(trialIndicesForN[1:])

        rw = RPLRaw()
        self.data = rw.data 
        self.analogTime = [i / sRate for i in range(len(self.data))]

        if plot_type == 'Signal':
            data = self.data[idx[0]:idx[-1]]
            if plotOpts['RemoveLineNoise']:
                data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
            x = np.linspace(-plotOpts['PreTrial'], 0, num = plotOpts['PreTrial'])
            x = np.concatenate((x, np.linspace(0, len(data) - plotOpts['PreTrial'], num = len(data) - plotOpts['PreTrial'])))
            ax.plot(x, data)
            ax.axvline(0, color = 'g') # Start of trial. 
            ax.axvline((self.timeStamps[i][1] - self.timeStamps[i][0]) * 30000, color = 'm')
            ax.axvline((self.timeStamps[i][2] - self.timeStamps[i][0]) * 30000, color = 'r')

        elif plot_type == 'FreqPlot':
            if plotOpts['PlotAllData']:
                data = self.data 
            else: 
                data = self.data[idx[0]:idx[-1]]
            if plotOpts['RemoveLineNoise']:
                data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
            datam = np.mean(data)
            fftProcessed, f = plotFFT(data - datam, sRate)
            ax.plot(f, fftProcessed)
            if plotOpts['LogPlot']:
                ax.set_yscale('log')

        elif plot_type == 'TFftt': 
        	if plotOpts['PlotAllData']:
        		dIdx = self.trialIndices[:, -1] - self.trialIndices[:, 0]
        		mIdx = np.amax(dIdx)
        		spTimeStep = plotOpts['TFfttWindow'] - plotOpts['TFfftOverlap']
        		spTimeBins = np.floor(mIdx/spTimeStep) - plotOpts['TFfftOverlap']/spTimeStep
        		nFreqs = (plotOpts['TFfftPoints'] / 2) + 1
        		ops = np.zeros(nFreqs, spTimeBins)
        		opsCount = ops 
        		for j in range(self.numSets):
        			tftIdx = self.trialIndices[j,:]
        			data = self.data[tftIdx[0]:tftIdx[-1]]
		            if plotOpts['RemoveLineNoise']:
		                data = removeLineNoise(data, plotOpts['RemoveLineNoiseFreq'], sRate)
            		datam = np.mean(data)
            		window = np.hamming(plotOpts['TFfftWindow'])
            		[s, f, t, im] = plt.specgram(data - datam, window = window, NFFT = plotOpts['TFfftPoints'], noverlap = plotOpts['TFfftOverlap'], fs = sRate)
        		pass 

        	else: 
	            tIdx = self.trialIndices[i,:]
	            idx = [tIdx[0] - ((plotOpts['TFfftStart']+500)/1000*sRate), tIdx[0] - ((plotOpts['TFfftStart']+1)/1000*sRate)]
	            data = self.data[idx[0]:idx[-1]]
	            datam = np.mean(data)
	            window = np.hamming(plotOpts['TFfftWindow'])
            	[s, f, t, im] = plt.specgram(data - datam, window = window, NFFT = plotOpts['TFfftPoints'], noverlap = plotOpts['TFfftOverlap'], fs = sRate)
            # data = obj.data.analogData(idx);
            # datam = mean(data);
            # [~,~,~,P]=spectrogram(data-datam,Args.TFfftWindow,Args.TFfftOverlap,Args.TFfftPoints,sRate,'yaxis');
            
            # %     Normalization parameters of the NP
            # Pmean=mean(P,2); %mean power density of each frequency bin
            # Pstd=std(P,0,2); %standard deviation of each frequency bin

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
