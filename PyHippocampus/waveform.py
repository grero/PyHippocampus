import numpy as np 
import DataProcessingTools as DPT 
import os 
import matplotlib.pyplot as plt 
import hickle as hkl
from .arrayplot import ArrayPlot

class Waveform(DPT.DPObject, ArrayPlot):
    """
    Please run this on the mountains directory under day level
    
    """
    filename = "waveform.hkl"
    argsList = [("loadFrom", None), ("mountainsDirectory", "mountains"), 
        ("ouputDirectory","output"), ("templateFilename","templates.hkl")]
    level = 'channel'

    def __init__(self, *args, **kwargs):
        fname = kwargs.get("loadFrom", None)
        if fname is not None:
            DPT.DPObject.__init__(self, *args, **kwargs)
        else:
            rr = DPT.levels.resolve_level(self.level, os.getcwd())
            with DPT.misc.CWD(rr):
                DPT.DPObject.__init__(self, *args, **kwargs)

        ArrayPlot.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        # thie function will be called by PanGUI.main once to create this waveform object
        ArrayPlot.create(self, *args, **kwargs)
        self.read_templates()  # read the mountainsort template files
        self.previous_plot_type = ''  # store the previous plot type to know if the ploty_type changes
        self.channel_idx = 0  
        self.array_idx = 0
        
        if self.data[0].all():
            # create object if data is not empty
            DPT.DPObject.create(self, *args, **kwargs)
        else:
            # create empty object if data is empty
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)            
        return self 

    def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False,\
             getPlotOpts = False, overlay = False, **kwargs):
        # this function will be called in different instances in PanGUI.main
        # eg. initially creating the window, right-clicking on the axis and click on any item
        # input argument:   'i' is the current index in the data list to plot 
        #                   'ax' is the axis to plot the data in
        #                   'getNumEvents' is the flag to get the total number of items and the current index of the item to plot, which is 'i'
        #                   'getLevels' is the flag to get the level that the object is supposed to be created in
        #                   'getPlotOpts' is the flag to get the plotOpts for creating the menu once we right-click the axis in the figure
        #                   'kwargs' is the keyward arguments pairs to update plotOpts
        
        # plotOpts is a dictionary to store the information that will be shown 
        # in the menu evoked by right-clicking on the axis after the window is created by PanGUI.create_window
        # for more information, please check in PanGUI.main.create_menu
        plotOpts = {'LabelsOff': False, 'TitleOff': False, \
                    'Type': DPT.objects.ExclusiveOptions(['channel', 'array'], 0)}

        self.update_local_plotopts(plotOpts, kwargs)  # update the plotOpts based on kwargs, this line is important to receive the input arguments and act accordingly
                    
        plot_type = plotOpts['Type'].selected()  # this variable will store the selected item in 'Type'
        
        if not self.previous_plot_type:  # initial assignement of self.previous_plot_type
            self.previous_plot_type = plot_type

        if getPlotOpts:  # this will be called by PanGUI.main to obtain the plotOpts to create a menu once we right-click on the axis
            return plotOpts 

        if getLevels:  # this will be called by PanGUI.main to the level that this object is supposed to be created in
            return ['channel', 'trial']
            

        if getNumEvents:  # this will be called by PanGUI.main to return two values: first value is the total number of items to pan through, second value is the current index of the item to plot
            if plot_type == self.previous_plot_type:  # no changes of plot_type
                if plot_type == 'channel':
                    return len(ArrayPlot.get_all_elements(self, 'channel')), i
                elif plot_type == 'array':
                    return len(ArrayPlot.get_all_elements(self, 'array')), i
            
            elif self.previous_plot_type == 'array' and plot_type == 'channel':  # change from array to channel
                self.previous_plot_type = 'channel'
                return len(ArrayPlot.get_all_elements(self, 'channel')), ArrayPlot.get_first_channel(self, i)
                    
            elif self.previous_plot_type == 'channel' and plot_type == 'array':  # change from channel to array
                self.previous_plot_type = 'array'
                return len(ArrayPlot.get_all_elements(self, 'array')), ArrayPlot.get_array_idx(self, i)
                

        if ax is None:
            ax = plt.gca()

        if not overlay:
            ax.clear()

        fig = ax.figure  # get the parent figure of the ax
        for x in fig.get_axes():  # remove all axes in current figure
            x.remove()
            
        ######################################################################
        #################### start plotting ##################################
        ######################################################################
        if plot_type == 'channel':  # plot in channel level
            y = self.data[i]
            x = np.arange(y.shape[0])
            ax = fig.add_subplot(111)
            ax.plot(x, y)
    
            ########labels###############
            if not plotOpts['TitleOff']:  # if TitleOff icon in the right-click menu is clicked
                ax.set_title(self.dirs[i])
                
            if not plotOpts['LabelsOff']:  # if LabelsOff icon in the right-click menu is clicked
                ax.set_xlabel('Time (sample unit)')
                ax.set_ylabel('Voltage (uV)')
            
        elif plot_type == 'array':  # plot in array level
            ArrayPlot.plot(self, i, fig, plotOpts)
            
        return ax
    
    
    def append(self, wf):
        # this function will be called by PanGUI.main to append the values of certain fields
        # from an extra object (wf) to this object
        # It is useful to store the information of the objects for panning through in the future
        DPT.DPObject.append(self, wf)  # append self.setidx and self.dirs
        ArrayPlot.append(self, wf)
        self.data = self.data + wf.data
        self.numSets += wf.numSets
        
    
    
    #%% helper functions        
    def read_templates(self):
        self.numSets = 1
        # make the following items as lists for the sake of self.append
        template_fileanme = os.path.join('..', '..', '..', 'mountains',
                                         self.channel_filename[0], 'output', 'templates.hkl')
        if os.path.isfile(template_fileanme):
            self.data = [np.squeeze(hkl.load(template_fileanme))]
        else:
            print('No mountainsort template file is found for {0}...'.format(self.channel_filename[0]))
            self.data = [np.array([])]
            
    def update_local_plotopts(self, plotOpts, kwargs):
        for (k, v) in plotOpts.items():
                    plotOpts[k] = kwargs.get(k, v)
    
