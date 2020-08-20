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
        self.previous_plot_type = ''
        self.channel_idx = 0
        self.array_idx = 0
        
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
        
        if not self.previous_plot_type:
            self.previous_plot_type = plot_type

        if getPlotOpts:
            return plotOpts 

        if getLevels:
            return ['channel', 'trial']
            

        if getNumEvents:
            if plot_type == self.previous_plot_type:  # no changes of plot_type
                if plot_type == 'channel':
                    return self.get_num_elements('channel'), i
                elif plot_type == 'array':
                    return self.get_num_elements('array'), i
            
            elif self.previous_plot_type == 'array' and plot_type == 'channel':  # change from array to channel
                self.previous_plot_type = 'channel'
                return self.get_num_elements('channel'), self.get_first_channel(i)
                    
            elif self.previous_plot_type == 'channel' and plot_type == 'array':  # change from channel to array
                self.previous_plot_type = 'array'
                return self.get_num_elements('array'), self.get_array_idx(i-2)
                

        if ax is None:
            ax = plt.gca()

        if not overlay:
            ax.clear()

        fig = ax.figure
        for x in fig.get_axes():  # remove all axes in current figure
            x.remove()
                
        #################### start plotting ##################################
        if plot_type == 'channel':
            y = self.data[i]
            x = np.arange(y.shape[0])
            ax = fig.add_subplot(111)
            ax.plot(x, y)
    
        ########labels###############
            if not plotOpts['TitleOff']:
                ax.set_title(self.channelFilename[i])
                
            if not plotOpts['LabelsOff']:
                ax.set_xlabel('Time (sample unit)')
                ax.set_ylabel('Voltage (uV)')
            
        elif plot_type == 'array':
            channel_idx, array_name = self.get_channel_idx(i)  # get the channels that belong to the same array
            num_channels = len(channel_idx)
            num_row, num_col = self.get_factors(num_channels)
            for k, x in enumerate(channel_idx):
                ax = fig.add_subplot(num_row, num_col, k+1)
                ax.plot(self.data[x])   
    
        ########labels###############
                if not plotOpts['TitleOff']:
                    ax.set_title('array{0:02}_{1}'.format(i+1, self.channelFilename[channel_idx[k]]))
                    
                if not plotOpts['LabelsOff']:
                    if num_col % (k+1) == 1 or k+1 == 1:
                        ax.set_ylabel('Voltage (uV)')
                    if k+1 >= num_col * (num_row-1):
                        ax.set_xlabel('Time (sample unit)')
                    

        
            
        return ax
    
    
    def append(self, wf):
        DPT.DPObject.append(self, wf)
        self.data = self.data + wf.data
        self.channelFilename = self.channelFilename + wf.channelFilename
        self.numSets += wf.numSets
        
    
    
    #%% helper functions        
    def get_first_channel(self, i):
        for k, x in enumerate(self.dirs):
            channel_temp = self.get_channels(i, x)
            if channel_temp:
                return  k  # return the first available channel of the array
                
    
    def get_num_elements(self, elements):
        array_idx_all = []
        for x in self.dirs:
            array_idx_all.append(int(re.search('(?<={0})\d+'.format(elements), x)[0]))
        return len(set(array_idx_all))
        
    def get_array_idx(self, i):
        return int(re.search('(?<=array)\d+', self.dirs[i])[0])
        
    def get_channel_idx(self, i):
        array_name = re.search('array\d+', self.dirs[i])[0]
        channel_idx = []
        for k, x in enumerate(self.dirs):
            if self.get_channels(i, x):
                channel_idx.append(k)
        return channel_idx, array_name
    
    def get_channels(self, i, x):
        if re.search('array(0)?{0}'.format(i+1), x):
            return int(re.search('(?<=channel)\d+', x)[0])
        else:
            return None
        
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
