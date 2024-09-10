#!/usr/bin/env python3
import os
import numpy as np
import spikeinterface.full as si
from spikeinterface_gui import MainWindow, mkQApp
import json
import csv
from mountainlab_pytools import mdaio
import sys

class MountainSortAnalyzer():
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

    def create_spiketrains(self):
        # read the spikes.npy file from either if the curated folder, if it exists,
        # or the uncurated one
        curated_folder = "curated_sorting_analyzer"
        if os.path.isdir(curated_folder):
            spikes_file = os.path.join(curated_folder,"sorting","spikes.npy")
        else:
            spikes_file = os.path.join(self.folder,"sorting","spikes.npy")

        spikes = np.load(spikes_file)
        units = {}
        for sp in spikes:
            unit_id = sp[1]
            # convert from acquistion units to miliseconds
            sptime = sp[0]/(self.sampling_rate/1000.0)
            if not unit_id in units:
                units[unit_id] = []
            units[unit_id].append(sptime)

        # create cell directories
        for unit_id,sptimes in units.items():
            cellname = "cell{:02d}".format(unit_id)
            if not os.path.isdir(cellname):
                os.mkdir(cellname)
            spfile = os.path.join(cellname, "spiketrain.csv") 
            with open(spfile,"w") as fid:
                writer = csv.writer(fid)
                print(type(sptimes))
                writer.writerows(sptimes)
                
    def apply_curation(self):
        curation_folder = "curated_sorting_analyzer"
        if not os.path.isdir(curation_folder):
            curation_json = os.path.join(self.folder, "spikeinterface_gui","curation_data.json")
            if os.path.isfile(curation_json):
                curation_dict = json.load(open(curation_json))
                if not "format_version" in curation_dict:
                    curation_dict["format_version"] = "1"

                clean_sorting_analyzer = si.apply_curation(self.analyzer, curation_dict=curation_dict)
                clean_sorting_analyzer.save_as(format="binary_folder", folder=curation_folder)

if __name__ == "__main__":
     app = mkQApp()
     mda_analyzer = MountainSortAnalyzer()
     win = MainWindow(mda_analyzer.analyzer, curation=mda_analyzer.curation)
     win.show()
     app.exec()
     # once the window closes, apply the curation and create the spike trains
     mda_analyzer.apply_curation()
     mda_analyzer.save_as_mda()
     #mda_analyzer.create_spiketrains()