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
    argsList = [("FilesOnly", False), ("DirsOnly", False)]
    level = "session"

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

    def append(self, df):
        # update fields in parent
        DPT.DPObject.append(self, df)
        # update fields in child
        self.itemList += df.itemList
        self.itemNum += df.itemNum

    def plot(self, i=None, getNumEvents=False, getLevels=False, 
             getPlotOpts=False, ax=None, **kwargs):
        """
        DirFiles.plot(Type=["Vertical", "Horizontal", "All"], BarWidth=0.8)
        """
        # set plot options
        plotopts = {"Type": DPT.objects.ExclusiveOptions(["Vertical", "Horizontal","All"],0),
                         "BarWidth": 0.8}
        if getPlotOpts:
            return plotopts
        
        # Extract the recognized plot options from kwargs
        for (k, v) in plotopts.items():
            plotopts[k] = kwargs.get(k, v)

        plottype = plotopts["Type"].selected()

        if getNumEvents:
            # Return the number of events available
            if plottype == "All":
                return 1, 0
            else:
                if i is not None:
                    nidx = i
                else:
                    nidx = 0
                return len(self.itemNum), nidx
            
        if getLevels:
            # Return the possible levels for this object
            return ["item", "all"]
        
        if ax is None:
            ax = gca()
        
        ax.clear()
            
        if plottype == "All":
            ax.bar(np.arange(len(self.itemNum)),self.itemNum, width=plotopts["BarWidth"])
        elif plottype == "Horizontal":
            ax.barh(1,self.itemNum[i],height=plotopts["BarWidth"])
        else:
            ax.bar(1,self.itemNum[i],width=plotopts["BarWidth"])
                
        return ax
