import os
from matplotlib.pyplot import gca
import DataProcessingTools as DPT
import numpy as np

class DirFiles(DPT.DPObject):
    """
    DirFiles(redoLevel=0, saveLevel=0, ObjectLevel='Session', 
             FilesOnly=False, DirsOnly=False)
    """
    filename = "dirfiles.hkl"
    argsList = [("FilesOnly", False), ("DirsOnly", False), ("ObjectLevel", "Session")]

    def __init__(self, *args, **kwargs):
        # initialize fields in parent
        DPT.DPObject.__init__(self, normpath=False, *args, **kwargs)

    def create(self, *args, **kwargs):
        # check for files or directories in current directory
        cwd = os.getcwd()
        dirList = os.listdir()
        
        if self.args["FilesOnly"]:
            print("Checking " + cwd + " for files")
            # filter and save only files
            itemList = list(filter(os.path.isfile, dirList))
        elif self.args["DirsOnly"]:
            print("Checking " + cwd + " for directories")
            # filter and save only dirs
            itemList = list(filter(os.path.isdir, dirList))
        else:
            print("Checking " + cwd + " for both files and directories")
            # save both files and directories
            itemList = dirList
            
        # check number of items
        dnum = len(itemList)
        print(str(dnum) + " items found")
        
        # create object if there are some items in this directory
        if dnum > 0:
            # update fields in parent
            self.dirs = [cwd]
            self.setidx = [0 for i in range(dnum)]
            # update fields in child
            self.itemList = itemList
            self.itemNum = [dnum]
            
        # set plot options
        self.plotopts = {"Type": DPT.objects.ExclusiveOptions(["Vertical", "Horizontal","All"],0),
                         "BarWidth": 0.8}
        
        # check if we need to save the object, with the default being 0
        if kwargs.get("saveLevel", 0) > 0:
            self.save()
        
    def append(self, df):
        # update fields in parent
        DPT.DPObject.append(self, df)
        # update fields in child
        self.itemList += df.itemList
        self.itemNum += df.itemNum

    def plot(self, i=None, ax=None, overlay=False):
        self.current_idx = i
        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()
            
        plottype = self.plotopts["Type"].selected()
        if plottype is "All":
            ax.bar(np.arange(len(self.itemNum)),self.itemNum, width=self.plotopts["BarWidth"])
        elif plottype is "Horizontal":
            ax.barh(1,self.itemNum[i],height=self.plotopts["BarWidth"])
        else:
            ax.bar(1,self.itemNum[i],width=self.plotopts["BarWidth"])
