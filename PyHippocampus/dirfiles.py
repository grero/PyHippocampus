import os
from matplotlib.pyplot import gca
import DataProcessingTools as DPT

class DirFiles(DPT.DPObject):
    def __init__(self, redoLevels=0, saveLevels=0, objectLevel='Session'):
        # initialize fields in parent
        DPT.DPObject.__init__(self)
        # check for files or directories in current directory
        dirListing = os.listdir()
        # check number of items
        dnum = len(dirListing)
        # create object if there are some items in this directory
        if dnum > 0:
            # update fields in parent
            self.dirs = [os.getcwd()]
            self.setidx = [0, dnum]
            # update fields in child
            self.dirList = dirListing
            self.dirNum = [dnum]

    def append(self, df):
        # update fields in parent
        DPT.DPObject.append(self, df)
        # update fields in child
        self.dirList += df.dirList
        self.dirNum += df.dirNum

    def plot(self, i, ax=None, overlay=False):
        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()
            
        ax.bar(1,self.dirNum[i],width=0.5)
