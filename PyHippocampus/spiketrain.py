import DataProcessingTools as DPT
import glob
import csv
import numpy as np
import os

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, suppress=True)

class Spiketrain(DPT.DPObject):
    filename = "spiketrain.hkl"
    argsList = []
    level = "cell"

    def __init__(self, *args, **kwargs):
        rr = DPT.levels.resolve_level("cell", os.getcwd())
        with DPT.misc.CWD(rr):
            DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        self.numSets = 0
        self.spiketimes = []

        csvFile = glob.glob('*.csv')
        if len(csvFile) == 0:
            kwargs["dirs"] = []
            print('No spiketrain file, creating empty object...')
            DPT.DPObject.create(self, *args, **kwargs)
        else: 
            DPT.DPObject.create(self, *args, **kwargs)
            with open('spiketrain.csv','r') as f:
                reader = csv.reader(f)
                c = []
                for row in reader:
                    c.append(row)
            self.spiketimes = [c[0]]
            self.numSets = 1 

    def append(self, st):
        # update fields in parent
        DPT.DPObject.append(self, st)
        # update fields in child
        self.numSets += st.numSets
        self.spiketimes += st.spiketimes


