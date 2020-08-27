import numpy as np
import re

class ArrayPlot:
    def __init__(self, *args, **kwargs):
        self.create_arrayplot(*args, **kwargs)
        
    def create_arrayplot(self, *args, **kwargs):
        """
        Public attributes
        """
        self.xlabelname = 'Time (sample unit)'
        self.ylabelname = 'Voltage (uV)'
        
        """
        Protected attributes
        """
        
        self._array_sets = [[1, 32],
                            [33, 64],
                            [65,  96],
                            [97, 124]]
        self._electrodes_array_size = [16,8]
        self._num_channel_per_array = 32
        self._electrodes_id = np.insert(np.arange(1, 125),
                                        [0,7,118,124], -1)
        
        for (k,v) in kwargs.items():
            if hasattr(self, k):
                self.__dict__[k] = v
            else:
                raise ValueError("{0} does not exist vmplot...".format(k))
        
    def plot_arrayplot(self, i, fig, plotOpts, *args, **kwargs):
        channel_idx, channel_locs = self.get_channels_in_array(i)  # get the channels that belong to the same array
        num_row, num_col, subplot_idx = self.get_subplots_grid(i, channel_idx)
        idx_title, idx_label = self.get_label_subplot(channel_idx)  # these outputs are the channel number to do the labelling
        ax = fig.subplots(num_row, num_col).flatten()
        for k, x in enumerate(subplot_idx):  # x will be the channel number
            if x == -1 or x not in channel_idx:  # empty channel
                ax[k].remove()
            else:
                idx_temp = channel_locs[channel_idx.index(x)]  # index in the entire channel list
                ax[k].plot(self.data[idx_temp])   
        
                """
                labels
                """
                if k//num_col != num_row-1:  # hide the x tick labels in all subplots except the last row
                    ax[k].get_xaxis().set_visible(False)
            
                if not plotOpts['TitleOff']:  # if TitleOff icon in the right-click menu is clicked
                    if x == idx_title:  # put array idx and channel idx as the title of the first subplot
                        ax[k].set_title('{0}\n{1}'.format(self.get_all_elements('array')[i],
                                                          self.channel_filename[idx_temp]))
                    else:  # only put the channel idx as the title for the rest
                        ax[k].set_title(self.channel_filename[idx_temp])
                    
            if not plotOpts['LabelsOff']:  # if LabelsOff icon in the right-click menu is clicked
                if x == idx_label:
                    ax[k].set_ylabel(self.ylabelname)
                    ax[k].set_xlabel(self.xlabelname)
                    
        return ax[0]
                    
                    
    #%% Helper functions
    def get_first_channel(self, i):
        target_str = self.get_all_elements('array')[i]
        for k, x in enumerate(self.dirs):
            if target_str in self.dirs[k]:
                return  k  # return the first available channel of the array
            
    def get_channels_in_array(self, i):
        target_str = self.get_all_elements('array')[i]
        channel_idx = []
        channel_locs = []
        for k, x in enumerate(self.dirs):
            if target_str in x:
                channel_idx.append(int(re.search('(?<=channel)\d+', x)[0]))
                channel_locs.append(k)
        return channel_idx, channel_locs
    
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
    def get_subplots_grid(self, i, channel_idx):
        num_col = self._electrodes_array_size[1]
        subplot_idx = [x if x in channel_idx else -1 for x in self._electrodes_id]
        
        start_idx = np.where(self._electrodes_id == self._array_sets[i][0])[0][0]
        end_idx = np.where(self._electrodes_id == self._array_sets[i][1])[0][0]  
        
        subplot_idx_start = start_idx // num_col * num_col
        subplot_idx_end = (end_idx // num_col + 1 ) * num_col
                                  
        subplot_idx = subplot_idx[subplot_idx_start: subplot_idx_end]
        
        num_row = (subplot_idx_end - subplot_idx_start) // num_col
        
        return num_row, num_col, subplot_idx
        
    #%% Labelling
    def get_label_subplot(self, channel_idx):
        """
        title will be in the first channel subplot
        label will be in the first column of the last row of the channel subplot
        """
        idx_title = channel_idx[0]
        
        idx_row = [np.where(self._electrodes_id == x)[0][0]//self._electrodes_array_size[1] for x in channel_idx]
        last_row = [channel_idx[k] for k, x in enumerate(idx_row) if x == max(idx_row)]
        idx_label = last_row[0]
        
        return idx_title, idx_label