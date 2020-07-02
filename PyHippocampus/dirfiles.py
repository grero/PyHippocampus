import os
from matplotlib.pyplot import gca
import DataProcessingTools as DPT
import h5py
import numpy as np
import hashlib

class DirFiles(DPT.DPObject):
    """
    DirFiles(redoLevel=0, saveLevel=0, objectLevel='Session', 
             filesOnly=0, dirsOnly=0)
    """
    filename = "dirfiles.h5"
    argsList = [("FilesOnly",0, "DirsOnly",0, "ObjectLevel","Session")]

    def __init__(self, *args, **kwargs):
        # initialize fields in parent
        DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        
        # check for files or directories in current directory
        dirList = os.listdir()
        print(dirList)
        
        if self.args["FilesOnly"]:
            # filter and save only files
            itemList = list(filter(os.path.isfile, dirList))
        elif self.args["DirsOnly"]:
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
            
        if kwargs.get("saveLevel",1) > 0:
            self.save()

    def save(self, fname=None):
        if fname is None:
            fname = self.get_filename()

        with h5py.File(fname, "w") as ff:
            args = ff.create_group("args")
            args["FilesOnly"] = self.args["FilesOnly"]
            args["DirsOnly"] = self.args["DirsOnly"]
            args["ObjectLevel"] = self.args["ObjectLevel"]
            ff["itemList"] = self.itemList
            ff["itemNum"] = self.itemNum
            ff["dirs"] = np.array(self.dirs, dtype='S256')
            ff["setidx"] = self.setidx

    def load(self, fname=None):
        DPT.DPObject.load(self)
        if fname is None:
            fname = self.filename
        with h5py.File(fname) as ff:
            args = {}
            for (k, v) in ff["args"].items():
                self.args[k] = v.value
            self.itemList = ff["itemList"][:]
            self.itemNum = ff["itemNum"][:]

    def hash(self):
        """
        Returns a hash representation of this object's arguments.
        """
        #TODO: This is not replicable across sessions
        h = hashlib.sha1(b"psth")
        for (k, v) in self.args.items():
            x = np.atleast_1d(v)
            h.update(x.tobytes())
        return h.hexdigest()

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
