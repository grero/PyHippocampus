import os
import numpy as np
from matplotlib.pyplot import gcf, gca
import DataProcessingTools as DPT
import PanGUI

class dirfiles(DPT.DPObject):
    def __init__(self, RedoLevels=0, SaveLevels=0, ObjectLevel='Session'):
        # initialize fields in parent
        DPT.DPObject.__init__(self)
        # check for files or directories in current directory
        dir_listing = os.listdir()
        # check number of items
        dnum = len(dir_listing)
        # create object if there are some items in this directory
        if dnum > 0:
            # update fields in parent
            self.dirs = [os.getcwd()]
            self.setidx = [0, dnum]
            # update fields in child
            self.dir_list = dir_listing

    def append(self, df):
        DPT.DPObject.append(self, df)
        self.dir_list = self.dir_list + df.dir_list
