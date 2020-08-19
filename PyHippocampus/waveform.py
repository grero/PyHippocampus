import numpy as np 
import DataProcessingTools as DPT 
import os 
import matplotlib.pyplot as plt 
import math
import re
from mountainlab_pytools import mdaio

class Waveform(DPT.DPObject):
    # Please run this on the mountains directory under day level
    filename = 'waveform.hkl'
    argsList = []
    level = 'channel'

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        self.read_templates()
        
        if self.data[0].all():
            # create object
            DPT.DPObject.create(self, *args, **kwargs)
        else:
            # create empty object
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)            
        return self 

    def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False,\
             getPlotOpts = False, overlay = False, **kwargs):

        plotOpts = {'LabelsOff': False, 'TitleOff': False, \
                    'Type': DPT.objects.ExclusiveOptions(['channel', 'array'], 1)}

        for (k, v) in plotOpts.items():
                    plotOpts[k] = kwargs.get(k, v)
                    
        plot_type = plotOpts['Type'].selected()

        if getPlotOpts:
            return plotOpts 

        if getLevels:
            return ['channel', 'trial']
            

        if getNumEvents:
            return self.numSets, i

        if ax is None:
            ax = plt.gca()

        if not overlay:
            ax.clear()

        #################### start plotting ##################################
        if plot_type == 'channel':
            y = self.data[i]
            x = np.arange(y.shape[0])
            ax.plot(x, y)
    
            if not plotOpts['TitleOff']:
                ax.set_title(self.channelFilename[i])
                
            if not plotOpts['LabelsOff']:
                ax.set_xlabel('Time (sample unit)')
                ax.set_ylabel('Voltage (uV)')
            
        elif plot_type == 'array':
            channel_idx, array_name = self.get_channel_idx(i)  # get the channels that belong to the same array
            num_channels = len(channel_idx)
            num_row, num_col = self.get_factors(num_channels)
            fig = ax.figure
            for x in fig.get_axes():
                x.remove()
            for i, x in enumerate(channel_idx):
                ax = fig.add_subplot(num_row, num_col, i+1)
                ax.plot(self.data[x])   
    
                if not plotOpts['TitleOff']:
                    ax.set_title('{0}_{1}'.format(array_name, self.channelFilename[channel_idx[i]]))
                    
                if not plotOpts['LabelsOff']:
                    if num_col % (i+1) == 1 or i+1 == 1:
                        ax.set_ylabel('Voltage (uV)')
                    if i+1 >= num_col * (num_row-1):
                        ax.set_xlabel('Time (sample unit)')
                    

        
            
        return ax
    
    
    def append(self, wf):
        DPT.DPObject.append(self, wf)
        self.data = self.data + wf.data
        self.channelFilename = self.channelFilename + wf.channelFilename
        self.numSets += wf.numSets
        
    
    
    #%% helper functions        
    def get_channel_idx(self, i):
        array_name = re.search('array\d+', self.dirs[i])[0]
        channel_idx = []
        for i, x in enumerate(self.dirs):
            if array_name in x:
                channel_idx.append(i)
        return channel_idx, array_name
        
    def get_factors(self, number):
        i = round(math.sqrt(number))
        while number % i:
            i -= 1
        return i, int(number/i)
        
    def read_templates(self):
        self.numSets = 1
        # make the following items as lists for the sake of self.append
        self.channelFilename = [os.path.basename(os.path.normpath(os.getcwd()))]
        self.data = [np.squeeze(mdaio.readmda(os.path.join('..', '..', '..', 'mountains',\
                                                          self.channelFilename[0], 'output', 'templates.mda')))]
