import numpy as np 
import DataProcessingTools as DPT 
from .rplparallel import RPLParallel
from .rpllfp import RPLLFP
from .helperfunctions import plotFFT, removeLineNoise
from .vmplot import VMPlot
import os 
import matplotlib.pyplot as plt 
from mountainlab_pytools import mdaio
import os

class WaveformPlot(DPT.DPObject):
    # Please run this on the mountains directory under day level
    filename = 'vmflp.hkl'
    argsList = []
    level = 'channel'

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        self.channelIdx = []
        self.data = []
        self.numSets = 0  
        
        self.read_templates()
        # templatesAll = DPT.levels.processLevel('channel', \
        #                                        "from mountainlab_pytools import mdaio;\
        #                                        import os;\
        #                                        print('{0}'.format(os.getcwd()));\
        #                                        mdaio.readmda(os.path.join('output', 'templates.mda'))")
        
        if self.data:
            # create object
            DPT.DPObject.create(self, *args, **kwargs)
            self.numSets = len(self.channelIdx)
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
            return ['trial', 'all']

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
            ax.set_title(self.channelIdx[i])
        else:
            ax.get_title().set_visible(False)
            
        if plot_type == 'channel':
            return ax;
        elif plot_type == 'array':
            return [ax, ax];
        else:
            raise(ValueError('Invalid plot_type in waveformplot...'))
    
    def read_templates(self):
        target_dir = 'mountains'
        for channel_dirs in os.listdir(target_dir):
            self.data.append(np.squeeze(mdaio.readmda(\
                                        os.path.join(target_dir, channel_dirs,\
                                                     'output', 'templates.mda'))))
            self.channelIdx.append(channel_dirs)
        
    
        
