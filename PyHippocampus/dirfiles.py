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
    argsList = [("filesOnly", False), ("dirsOnly", False)]
    level = "session"


    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)


    def create(self, *args, **kwargs):
        # check for files or directories in current directory
        cwd = os.getcwd()
        dirList = os.listdir()
        
        if self.args["filesOnly"]:
            print("Checking " + cwd + " for files")
            # filter and save only files
            itemList = list(filter(os.path.isfile, dirList))
        elif self.args["dirsOnly"]:
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
            # create object
            DPT.DPObject.create(self, *args, **kwargs)
            # update fields in child
            self.itemList = itemList
            self.itemNum = [dnum]
        else:
            # create empty object
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)


    def append(self, df):
        # update fields in parent
        DPT.DPObject.append(self, df)
        # update fields in child
        self.itemList += df.itemList
        self.itemNum += df.itemNum


    def plot(self, i=None, getNumEvents=False, getLevels=False, 
             getPlotOpts=False, ax=None, preOpt=None, **kwargs):
        """
        DirFiles.plot(PlotType=["Vertical", "Horizontal", "All"], BarWidth=0.8)
        """
        # set plot options
        plotopts = {"PlotType": DPT.objects.ExclusiveOptions(["Vertical", "Horizontal","All"],0),
                         "BarWidth": 0.8}
        if getPlotOpts:
            return plotopts
        
        # Extract the recognized plot options from kwargs
        for (k, v) in plotopts.items():
            plotopts[k] = kwargs.get(k, v)

        plottype = plotopts["PlotType"].selected()

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
