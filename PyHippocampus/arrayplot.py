import numpy as np
import re

class ArrayPlot:
    def __init__(self, *args, **kwargs):
        self.create_arrayplot(*args, **kwargs)
        
    def create_arrayplot(self, *args, **kwargs):
        self.xlabelname = 'Time (sample unit)'
        self.ylabelname = 'Voltage (uV)'
        
        for (k,v) in kwargs.items():
            if hasattr(self, k):
                self.__dict__[k] = v
            else:
                raise ValueError("{0} does not exist vmplot...".format(k))
        
    def plot_arrayplot(self, i, fig, plotOpts, *args, **kwargs):
        channel_idx = self.get_channels_in_array(i)  # get the channels that belong to the same array
        num_channels = len(channel_idx)
        num_row, num_col = self.get_subplots_grid(num_channels)
        for k, x in enumerate(channel_idx):
            ax = fig.add_subplot(num_row, num_col, k+1)
            ax.plot(self.data[x])   
        
        ########labels###############
            if k//num_col != num_row-1:  # hide the x tick labels in all subplots except the last row
                ax.get_xaxis().set_visible(False)
        
            if not plotOpts['TitleOff']:  # if TitleOff icon in the right-click menu is clicked
                if k == 0:  # put array idx and channel idx as the title of the first subplot
                    ax.set_title('{0}\n{1}'.format(self.get_all_elements('array')[i], \
                                                   self.channel_filename[channel_idx[k]]))
                else:  # only put the channel idx as the title for the rest
                    ax.set_title(self.channel_filename[channel_idx[k]])
                
            if not plotOpts['LabelsOff']:  # if LabelsOff icon in the right-click menu is clicked
                if k // num_col == num_row-1 and k % num_col == 0:  # last row and first column
                    ax.set_ylabel(self.ylabelname)
                    ax.set_xlabel(self.xlabelname)
                    
        return ax
                    
                    
    #%% Helper functions
    def get_first_channel(self, i):
        target_str = self.get_all_elements('array')[i]
        for k, x in enumerate(self.dirs):
            if target_str in self.dirs[k]:
                return  k  # return the first available channel of the array
            
    def get_channels_in_array(self, i):
        target_str = self.get_all_elements('array')[i]
        channel_idx = []
        for k, x in enumerate(self.dirs):
            if target_str in x:
                channel_idx.append(k)
        return channel_idx
    
    def get_all_elements(self, elements):
        array_idx_all = []
        for x in self.dirs:
            array_idx_all.append(self.get_fullname(x, elements))
        return np.sort([x for x in set(array_idx_all)])
    
            
    def get_array_idx(self, i):
        target_str = self.get_fullname(self.dirs[i], 'array')
        for k, x in enumerate(self.get_all_elements('array')):
            if target_str in x:
                return k
        
    def get_fullname(self, x, elements):
        return x[:re.search('{0}\d+'.format(elements), x).span()[1]]
    
    #%% Subplot layout
    def get_subplots_grid(self, number):
        num_row = np.floor(np.sqrt(number))
        num_col = np.ceil(np.sqrt(number))
        return num_row, num_col
        