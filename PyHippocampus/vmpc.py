from .rplparallel import RPLParallel 
from .spiketrain import Spiketrain
from .umaze import Umaze
import numpy as np 
import DataProcessingTools as DPT 
import os 
import time
import matplotlib.pyplot as plt 
import re


class VMPlaceCell(DPT.DPObject):

    filename = "vmplacecell.hkl"
    argsList = [("NumShuffles", 100), ("ShuffleBounds", [0.1, 0.9]),
        ('GridSteps', 40), ('OverallGridSize', 25), ('MinObs', 5)]
    level = "cell"

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        
        self.numSets = 0
        
        um = Umaze(*args, **kwargs)
        st = Spiketrain(*args, **kwargs)
        rp = RPLParallel(*args, **kwargs)
        
        if len(um.sessionTime) > 0 and len(st.spiketimes) > 0 and len(rp.timeStamps) > 0: # able to get needed objects
            
            # create object
            print('Creating VMPlaceCell')
            DPT.DPObject.create(self, *args, **kwargs)
            
            stimes = np.array(st.spiketimes[0]).astype(float)
            stedges = um.sessionTime[:,0]
            umst = um.sessionTime
            
            sicValues_out2 = []
            rateMap_out2 = []
        
            for shuf in range(self.args['NumShuffles']+1):
                
                internal_time = time.time()
                
                if shuf % 100 == 0:
                    print(shuf)
                
                rplend = max([rp.timeStamps[-1][-1], stimes[-1]])
                
                # initialize array to store all spike times (original and shuffled)
                shuffled_train = stimes
                
                # generate random shifting amount (bounded by argument)
                time_shift = np.random.random()
                time_shift = rplend*(self.args['ShuffleBounds'][1] - self.args['ShuffleBounds'][0])*time_shift + self.args['ShuffleBounds'][0]*rplend
                
                if shuf == 0:
                    time_shift = 0
                
                # shift original spike times by random amount determined above
                shuffled_train = np.add(shuffled_train, time_shift)
                
                # wrap spike times that exceed final ripple timing to start
                shuffled_train[shuffled_train > rplend] = shuffled_train[shuffled_train > rplend] - rplend;        
                
                # print('time shifted, ', time.time() - internal_time)
                internal_time = time.time()
                
                # bin edges defined, binned into sessionTime intervals
                binned = np.histogram(shuffled_train/1000, stedges)
                binned = binned[0]
                
                # print('binned in, ', time.time() - internal_time)
                internal_time = time.time()
                
                # attribute spikes in each interval to the correct bin number (1-1600)
                # at the same time, durations are collated
                spike_count = np.zeros((1601,1))
                bin_durs = np.zeros((1601,1))
                for row in range(len(binned)):
                    spike_count[np.int64(umst[row,1])] = spike_count[np.int64(umst[row,1])] + binned[row]
                    bin_durs[np.int64(umst[row,1])] = bin_durs[np.int64(umst[row,1])] + umst[row,2]
                spike_count = spike_count[1:]
                bin_durs = bin_durs[1:]
                
                # print('attributed, ', time.time() - internal_time)
                internal_time = time.time()
                
                if shuf == 0:
                    rateMap_out2 = spike_count / bin_durs
                
                sic_values = self.spatialinfo(spike_count, bin_durs)
            
                # print('sic calculated, ', time.time() - internal_time)
                internal_time = time.time() 
            
                sicValues_out2.append(sic_values[0])
            
            self.numSets = 1
            self.sicValues = [sicValues_out2]
            self.rateMap = [np.squeeze(rateMap_out2)]
            
        else:
            # create empty object
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)  
            
        return self

    def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False,\
             getPlotOpts = False, overlay = False, **kwargs):

        plotOpts = {'SICOff': False, \
                    'Type': DPT.objects.ExclusiveOptions(['cell', 'channel'], 0)}

        self.update_local_plotopts(plotOpts, kwargs)  # update the plotOpts based on kwargs, this line is important to receive the input arguments and act accordingly
                    
        plot_type = plotOpts['Type'].selected()  # this variable will store the selected item in 'Type'
        
        try:
            if not self.previous_plot_type:  # initial assignement of self.previous_plot_type
                self.previous_plot_type = plot_type
        except:
            self.previous_plot_type = plot_type

        if getPlotOpts:  # this will be called by PanGUI.main to obtain the plotOpts to create a menu once we right-click on the axis
            return plotOpts 

        if getLevels:  # this will be called by PanGUI.main to the level that this object is supposed to be created in
            return ['channel', 'trial']
            

        if getNumEvents:  

            if plot_type == self.previous_plot_type:  # no changes of plot_type
                #print(len(self.get_all_elements(plot_type)), i)
                return len(self.get_all_elements(plot_type)), i
            
            elif self.previous_plot_type == 'channel' and plot_type == 'cell':  # change from channel to cell
                self.previous_plot_type = 'cell'
                i = self.get_granular_idx(i)[0]
                #print(len(self.get_all_elements('cell')), self.get_idx(i, 'cell'))
                return len(self.get_all_elements('cell')), self.get_idx(i, 'cell')
                    
            elif self.previous_plot_type == 'cell' and plot_type == 'channel':  # change from cell to channel
                self.previous_plot_type = 'channel'
                #print(len(self.get_all_elements('channel')), self.get_idx(i, 'channel'))
                return len(self.get_all_elements('channel')), self.get_idx(i, 'channel')  
            
            return 
                

        if ax is None:
            ax = plt.gca()

        if not overlay:
            ax.clear()
            
        fig = ax.figure  # get the parent figure of the ax
        for x in fig.get_axes():  # remove all axes in current figure
            x.remove()            
        
        if plot_type == 'cell':  # plot in cell level
            y = self.rateMap[i]
            y = np.reshape(y, (40, 40))
            ax = fig.add_subplot(1,1,1)
            ax.imshow(y)   
            if not plotOpts['SICOff']:
                title = self.dirs[i] + '\n cutoff, actual: (' + str(round(np.percentile(self.sicValues[i][1:], 95),4)) + ', ' + str(round(self.sicValues[i][0],4)) + ')'
            else:
                title = self.dirs[i]
            ax.set_title(title)
    
        else: # plot in channel level
            
            to_plot = self.get_granular_idx(i)
            if len(to_plot) > 9:
                print('too many cells to plot')
                return
            
            subplot_dim = [1,1]
            change = 0
            while subplot_dim[0]*subplot_dim[1] < len(to_plot):
                subplot_dim[change] += 1
                if change == 1:
                    change = 0
                else:
                    change = 1
            
            ax = fig.subplots(subplot_dim[0], subplot_dim[1]).flatten()    
            
            for idx in range(len(ax)):
                
                if idx >= len(to_plot):
                    ax[idx].remove()
                    
                else:
                    y = self.rateMap[to_plot[idx]]
                    y = np.reshape(y, (40, 40))
    
                    ax[idx].imshow(y)
                    
                    if not plotOpts['SICOff']:
                        title = self.dirs[to_plot[idx]] + '\n cutoff, actual: (' + str(round(np.percentile(self.sicValues[to_plot[idx]][1:], 95),4)) + ', ' + str(round(self.sicValues[to_plot[idx]][0],4)) + ')'
                    else:
                        title = self.dirs[to_plot[idx]]
                    
                    ax[idx].set_title(title)                
            
        return ax
    
    def append(self, pc):
        
        DPT.DPObject.append(self, pc)  # append self.setidx and self.dirs
        self.numSets += pc.numSets
        self.sicValues += pc.sicValues
        self.rateMap += pc.rateMap


    # computation helpers

    def spatialinfo(self, spike_maps, dur_map):
        
        # compute occupancy ratio per grid (1 x num_grids)
        dur_ratio = dur_map/sum(dur_map)
        
        # compute firing rate per grid, per shuffle (num_grids x num_iter)
        rate_maps = spike_maps / dur_map
        
        # compute average firing rate, per shuffle (1 x num_iter)
        average_fr = dur_ratio * rate_maps
        average_fr = np.nansum(average_fr, 0)
        
        # compute relative firing rate per grid, per shuffle (num_grids x num_iter)
        fr_ratio = rate_maps / average_fr
    
        # sic value, following formula, summed over all bins (1 x num_iter)
        sic_values = dur_ratio * fr_ratio * np.log2(fr_ratio)
        sic_values = np.nansum(sic_values, 0)
        
        return sic_values


    # plotting helpers

    def update_local_plotopts(self, plotOpts, kwargs):
        for (k, v) in plotOpts.items():
                    plotOpts[k] = kwargs.get(k, v)
            
    def get_idx(self, i, level):
        target_str = self.get_fullname(self.dirs[i], level)
        for k, x in enumerate(self.get_all_elements(level)):
            #print(target_str, x, k)
            if target_str in x:
                return k
        
    def get_fullname(self, x, elements):
        return x[:re.search('{0}\d+'.format(elements), x).span()[1]]
    
    def get_all_elements(self, elements):
        array_idx_all = []
        for x in self.dirs:
            array_idx_all.append(self.get_fullname(x, elements))
        return np.sort([x for x in set(array_idx_all)])  
    
    def get_granular_idx(self, i):
        target_str = self.get_all_elements('channel')[i]  
        output = []
        for idx in range(len(self.dirs)):
            if self.get_fullname(self.dirs[idx], 'channel') == target_str:
                output.append(idx)
        #print(output)
        return output
        
        
        