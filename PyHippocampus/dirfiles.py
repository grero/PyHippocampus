import os
from matplotlib.pyplot import gca
import DataProcessingTools as DPT

class DirFiles(DPT.DPObject):
    """
    DirFiles(redoLevel=0, saveLevel=0, ObjectLevel='Session', 
             FilesOnly=0, DirsOnly=0)
    """
    filename = "dirfiles.hkl"
    argsList = [("FilesOnly", False), ("DirsOnly", False), ("ObjectLevel", "Session")]

    def __init__(self, *args, **kwargs):
        # initialize fields in parent
        DPT.DPObject.__init__(self, normpath=False, *args, **kwargs)

    def create(self, *args, **kwargs):
        saveLevel = kwargs.get("saveLevel", 0)
        # check for files or directories in current directory
        dirList = os.listdir()
        print(dirList)
        
        if self.args["FilesOnly"]:
            print("FilesOnly")
            # filter and save only files
            itemList = list(filter(os.path.isfile, dirList))
        elif self.args["DirsOnly"]:
            print("DirsOnly")
            # filter and save only dirs
            itemList = list(filter(os.path.isdir, dirList))
        else:
            # save both files and directories
            itemList = dirList
            
        # check number of items
        dnum = len(itemList)
        print(dnum)
        
        # create object if there are some items in this directory
        if dnum > 0:
            # update fields in parent
            self.dirs = [os.getcwd()]
            self.setidx = [0 for i in range(dnum)]
            # update fields in child
            self.itemList = itemList
            self.itemNum = [dnum]
            
        if saveLevel > 0:
            self.save()
        
    def append(self, df):
        # update fields in parent
        DPT.DPObject.append(self, df)
        # update fields in child
        self.itemList += df.itemList
        self.itemNum += df.itemNum

    def plot(self, i, ax=None, overlay=False):
        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()
            
        ax.bar(1,self.itemNum[i],width=0.5)
