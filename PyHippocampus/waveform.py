import numpy as np 
import DataProcessingTools as DPT 
import os 
import matplotlib.pyplot as plt 
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
                    'Type': DPT.objects.ExclusiveOptions(['channel'], 0)}

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
            if type(ax).__module__ == np.__name__:
                for x in np.reshape(ax.size,1):
                    x[0].clear()
            else:
                ax.clear()

        #################### start plotting ##################################
        if plot_type == 'channel':
            y = self.data[i]
            x = np.arange(y.shape[0])
            ax.plot(x, y)

        if not plotOpts['LabelsOff']:
            ax.set_xlabel('Time (sample unit)')
            ax.set_ylabel('Voltage (uV)')
        else:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if not plotOpts['TitleOff']:
            ax.set_title(self.channelFilename[i])
        else:
            ax.get_title().set_visible(False)
            
        return ax
    
    
    def append(self, wf):
        self.data = self.data + wf.data
        self.channelFilename = self.channelFilename + wf.channelFilename
        self.numSets += wf.numSets
    
    
    #%% helper functions        
    def read_templates(self):
        self.numSets = 1
        # make the following items as lists for the sake of self.append
        self.channelFilename = [os.path.basename(os.path.normpath(os.getcwd()))]
        self.data = [np.squeeze(mdaio.readmda(os.path.join('..', '..', '..', 'mountains',\
                                                          self.channelFilename[0], 'output', 'templates.mda')))]
