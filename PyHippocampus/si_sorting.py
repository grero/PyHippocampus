#!/usr/bin/env python3
import os
import numpy as np
import spikeinterface.full as si
from spikeinterface_gui import MainWindow, mkQApp
import json
import csv
from mountainlab_pytools import mdaio
import sys
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import argparse

# nice colors than matplotlib's default
wong_colors = [(0.0,0.44705883,0.69803923),
               (0.9019608,0.62352943,0.0),
               (0.0,0.61960787,0.4509804),
               (0.8,0.4745098,0.654902),
               (0.3372549,0.7058824,0.9137255),
               (0.8352941,0.36862746,0.0),
               (0.9411765,0.89411765,0.25882354)]

other_colors = mpl.colormaps['Paired'].colors

def find_analyzers(basedir="."):
    """
    Find all curated sorting analyzers under `basedir`
    """
    paths = []
    for root, dirs, filenames in os.walk(basedir):
        for d in dirs:
            if d == "curated_sorting_analyzer":
                paths.append(os.path.join(root, d))
    return paths

def plot_analyzers(basedir="."):
    paths = find_analyzers(basedir)
    for path in paths:
        analyzer = si.load_sorting_analyzer(path)
        fig = plot_summary(analyzer)
        if fig == None:
            continue
        bb,qq = os.path.split(path)
        fig.savefig(os.path.join(bb, "curated_sorting_analyzer.pdf"))

def plot_summary(analyzer,save=True):
    n_units = analyzer.get_num_units()
    if n_units == 0:
        return None
    unit_ids = analyzer.sorting.get_unit_ids()
    if n_units < len(wong_colors):
        unit_colors = {k: wong_colors[i] for i,k in enumerate(unit_ids)}
    else:
        unit_colors = {k: other_colors[i] for i,k in enumerate(unit_ids)}
    fig = plt.figure(layout="constrained")
    fig.set_size_inches((9.7, 6.8))
    nrows = 1+n_units
    ncols = 1+n_units
    wr = [1 for i in range(n_units)]
    wr.append(3)
    gs = GridSpec(nrows, ncols, width_ratios=wr, figure=fig)
    axes = [fig.add_subplot(gs[0,j]) for j in range(n_units)]
    si.plot_unit_templates(sorting_analyzer_or_templates=analyzer, axes=axes, unit_colors=unit_colors)
    # post-process; equalize axis
    ylims = (np.inf, -np.inf)
    for ax in axes:
        _ylims = ax.get_ylim()
        ylims = (min(ylims[0], _ylims[0]), max(ylims[1], _ylims[1]))
    for ax in axes:
        ax.set_ylim(*ylims)
    axes2 = [fig.add_subplot(gs[1+i, j]) for i in range(n_units) for j in range(n_units)]
    axes2 = np.asarray(axes2).reshape(n_units, n_units)
    si.plot_crosscorrelograms(sorting_analyzer_or_sorting=analyzer, axes=axes2)
    # post process; don's show axis titles
    for ax in axes2[0,:]:
        ax.set_title(None)
    # add text describing these plots as cross-correlograms

    axes2[n_units//2,0].set_ylabel("Cross-correlograms")
    # plot scatter plot of features
    comps = analyzer.extensions["principal_components"].get_data()
    spikes = analyzer.extensions["random_spikes"].get_random_spikes()
    unit_idx = [sp[1] for sp in spikes]
    # plot_unit_templates should have created one axis for each unit idx
    ax = fig.add_subplot(gs[:, ncols-1])
    ax.scatter(comps[:,0], comps[:,1],c=[wong_colors[ii] for ii in unit_idx])
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    return fig

class MountainSortAnalyzer():
    level = "channel"
    def __init__(self, raw_data_file="dataset/raw_data.mda", firings_file="output/firings.mda",
                       sampling_rate=30000.0,redo=False, curation=True):
        self.sampling_rate = sampling_rate
        raw_data_dir, raw_data_file = os.path.split(raw_data_file)
        params_file = os.path.join(raw_data_dir, "params.json")
        if not os.path.isfile(params_file):
            with open(params_file, "w") as fid:
                dd = {'samplerate': sampling_rate}
                fid.write(json.dumps(dd))

        sorting = si.read_mda_sorting(firings_file, sampling_rate)
        recording = si.read_mda_recording(raw_data_dir,raw_fname=raw_data_file)
        self.curation = curation
        self.curated_analyzer = None
        # figure out the array based on the channel, and that we use 32 channels per array
        # note that should should change when we start using neuropixels
        rst,channel = os.path.split(os.getcwd())
        rst,_ = os.path.split(rst)
        rst,date = os.path.split(rst)
        self.date = date
        self.channel = int(channel[-3:])
        self.array = int(np.floor(self.channel/32))+1

        # create the analyzer

        self.folder = "sorting_analyzer"
        loaded = False
        if not redo and os.path.isdir(self.folder):
            try:
                self.analyzer = si.load_sorting_analyzer(self.folder)
                loaded = True
            except:
                loaded = False
                redo = True
        if not loaded:
            self.analyzer = si.create_sorting_analyzer(sorting=sorting,
                                                recording=recording,
                                                format="binary_folder",
                                                return_scaled=True, # this is the default to attempt to return scaled
                                                folder=self.folder,
                                                overwrite=redo,
                                                )
            # compute all the necessary components to do curation
            self.analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
            self.analyzer.compute("isi_histograms")
            self.analyzer.compute("waveforms")
            self.analyzer.compute("templates", operators=["average", "median", "std"])
            self.analyzer.compute("unit_locations")
            self.analyzer.compute("template_similarity")
            self.analyzer.compute("principal_components")
            self.analyzer.compute("correlograms")
            self.analyzer.compute("spike_amplitudes")

    def plot(self, curation=True):
        si.plot_sorting_summary(sorting_analyzer=self.analyzer, curation=curation,
                                backend='spikeinterface_gui')

    def plot_summary(self):
        print("Curated_analyzer: {}".format(self.curated_analyzer))
        if self.curated_analyzer is not None:
            fig = plot_summary(self.curated_analyzer)
        if fig is not None:
            bb,qq = os.path.split(self.folder)
            fig.savefig(os.path.join(bb, "curated_sorting_analyzer.pdf"))

    def save_as_mda(self):
        # only do this if we actually curated anything
        if os.path.isfile(os.path.join(self.folder,"spikeinterface_gui","curation_data.json")):
            curated_firings_file = "output/firings_curated.mda"
            if os.path.isfile(curated_firings_file):
                print("Curated firings already exist. Skipping...")
            else:
                curated_folder = "curated_sorting_analyzer"
                if os.path.isdir(curated_folder):
                    spikes_file = os.path.join(curated_folder,"sorting","spikes.npy")
                else:
                    spikes_file = os.path.join(self.folder,"sorting","spikes.npy")

                spikes = np.load(spikes_file)
                firings = np.zeros((3, len(spikes)))
                for i in range(len(spikes)):
                    firings[0,i] = spikes[i][2]+1
                    firings[1,i] = spikes[i][0]
                    firings[2,i] = spikes[i][1]+1

                mdaio.writemda(firings, curated_firings_file, dtype='float64')

    def get_start_indices(self):
        start_indices = {} 
        if os.path.isfile("start_times.mat"):
            _start_times = scipy.io.loadmat("start_times.mat")
            for jj in range(len(_start_times["start_indices"])):
                session_name =  _start_times["start_indices"][1][jj][0] 
                start_idx = _start_times["start_indices"][0][jj][0]-1 # because we are coming from matlab which uses 1-based indexing
                if jj == len(_start_times)-1: # the end
                    end_idx = None
                else:
                    end_idx = _start_times["start_indices"][0][jj+1][0]-1
                start_indices[session_name] = (start_idx,end_idx)
        else:
            start_indices = {"session01": (1,None)}
        return start_indices

    def get_cell_path(self, session, unit_id):
        unit_path = os.path.join("..", "..", session, 
                                 "array{:02d}".format(self.array),
                                "channel{:03d}".format(self.channel),
                                "cell{:02d}".format(unit_id))
        return unit_path

    def save_spiketrain(self, sptimes,unit_path):
        if not os.path.isdir(unit_path):
            os.makedirs(unit_path)
        else:
            r = input("{} already exists. Ovewrite y/n [n]?".format(unit_path))
            if r != "y":
                return None
        print("Writing cell to {}".format(unit_path))
        spfile = os.path.join(unit_path, "spiketrain.csv")
        with open(spfile, "w") as fid:
            writer = csv.writer(fid)
            writer.writerows(sptimes)
        
    def create_spiketrains(self,savelevel=1):
        # read the spikes.npy file from either if the curated folder, if it exists,
        # or the uncurated one
        start_indices = self.get_start_indices()

        if self.curated_analyzer is None:
            self.apply_curation(savelevel-1)

        unit_ids = self.curated_analyzer.sorting.get_unit_ids()
        for ii,uid in enumerate(unit_ids):
            spikes = self.curated_analyzer.sorting.get_unit_spike_train(uid)
            for sn,sidx in start_indices.items():
                sidx0 = np.searchsorted(spikes, sidx[0])[0]
                if sidx[1] is None:
                    sidx1 = len(spikes)-1
                else:
                    sidx1 = np.searchsorted(spikes, sidx[1])[0]
                # convert from acquistion units to miliseconds
                sptimes = [(spikes[ii]-sidx[0])/(self.sampling_rate/1000.0) for ii in range(sidx0,sidx1)]
                unit_path = self.get_cell_path(sn, ii+1)
                self.save_spiketrain(sptimes, unit_path)


    def apply_curation(self,savelevel=0):
        curation_folder = "curated_sorting_analyzer"
        curation_json = os.path.join(self.folder, "spikeinterface_gui","curation_data.json")
        if os.path.isfile(curation_json):
            curation_dict = json.load(open(curation_json))
            if not "format_version" in curation_dict:
                curation_dict["format_version"] = "1"

            clean_sorting_analyzer = si.apply_curation(self.analyzer, curation_dict=curation_dict)
            if savelevel > 0:
                if os.path.isdir(curation_folder):
                    shutil.move(curation_folder, "curation_folder_old")
                clean_sorting_analyzer.save_as(format="binary_folder", folder=curation_folder)
	     
            self.curated_analyzer = clean_sorting_analyzer

def parse_args():
    parser = argparse.ArgumentParser("Spike sorting curation")
    parser.add_argument("--features_only", action='store_true', help='Only compute features')
    parser.add_argument("--channels_file", type=str, help="A csv file with a list of channels to sort",
                        default="")
    parser.add_argument("--skip_error", action='store_true', help="Whether to skip channels with errors")
    parser.add_argument("--create_spiketrains", action='store_true', help="Create spiketrains from existing curated sorting")
    return parser.parse_args()

if __name__ == "__main__":
    arglist = parse_args()
    if arglist.channels_file:
        dirnames = []
        with open(arglist.channels_file,"r") as fid:
                reader = csv.reader(fid)
                for row in reader:
                    date = row[0]
                    # hack; the first row is the header
                    if row[1] == "Channel":
                        continue
                    channel = int(row[1])
                    dname = os.path.join(date,"mountains","channel{:03d}".format(channel))
                    if os.path.isdir(dname):
                        dirnames.append(dname)

        pwd = os.getcwd()
        for dname in dirnames:
            try:
                os.chdir(dname)
                mda_analyzer = MountainSortAnalyzer()
                if arglist.create_spiketrains:
                    mda_analyzer.apply_curation()
                    mda_analyzer.create_spiketrains()
            except Exception as ee:
                print(dname)
                if not arglist.skip_error:
                    raise ee
            finally:
                os.chdir(pwd)
    else:
        if not (arglist.features_only or arglist.create_spiketrains):
            mda_analyzer = MountainSortAnalyzer()
            app = mkQApp()
            win = MainWindow(mda_analyzer.analyzer, curation=mda_analyzer.curation)
            win.show()
            app.exec()
            # once the window closes, apply the curation and create the spike trains
            mda_analyzer.apply_curation()
            mda_analyzer.save_as_mda()
            mda_analyzer.plot_summary()
            #mda_analyzer.create_spiketrains()
        elif arglist.create_spiketrains:
            mda_analyzer = MountainSortAnalyzer()
            mda_analyzer.apply_curation()
            mda_analyzer.create_spiketrains()

    


