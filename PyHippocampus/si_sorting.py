import os
import numpy as np
import spikeinterface.full as si
from spikeinterface_gui import MainWindow, mkQApp
import json

class MountainSortAnalyzer():
    def __init__(self, raw_data_file="dataset/raw_data.mda", firings_file="output/firings.mda",
                       sampling_rate=30000.0,redo=False, curation=True):
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

        folder = "sorting_analyzer"
        if not redo and os.path.isdir(folder):
            self.analyzer = si.load_sorting_analyzer(folder)
        else:
            self.analyzer = si.create_sorting_analyzer(sorting=sorting,
                                                recording=recording,
                                                format="binary_folder",
                                                return_scaled=True, # this is the default to attempt to return scaled
                                                folder=folder,
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


if __name__ == "__main__":
     app = mkQApp()
     mda_analyzer = MountainSortAnalyzer()
     win = MainWindow(mda_analyzer.analyzer)
     win.show()
     app.exec()