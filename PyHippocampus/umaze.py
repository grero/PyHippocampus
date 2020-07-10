import DataProcessingTools as DPT
from pylab import gcf, gca
import numpy as np
import os
import glob
from . import unity


class Umaze(DPT.DPObject):
    filename = "umaze.hkl"
    argsList = []
    level = ""

    def __init__(self, *args, **kwargs):
            DPT.DPObject.__init__(self, normpath=False, *args, **kwargs)

    def create(self, *args, **kwargs):
        # set plot options
        self.indexer = self.getindex("trial")

        # load the unity object to get the data
        uf = unity.Unity()
        sumCost = uf.sumCost
        unityData = uf.unityData
        unityTriggers = uf.unityTriggers
        unityTrialTime = uf.unityTrialTime
        unityTime = uf.unityTime

        #

        # check if we need to save the object, with the default being 0
        if kwargs.get("saveLevel", 0) > 0:
            self.save()

    def plot(self, i=None, getNumEvents=False, getLevels=False, getPlotOpts=False, ax=None, **kwargs):
        # set plot options
        plotopts = {}
        if getPlotOpts:
            return plotopts

        # Extract the recognized plot options from kwargs
        for (k, v) in plotopts.items():
            plotopts[k] = kwargs.get(k, v)

        if ax is None:
            ax = gca()
        ax.clear()

        return ax

    def append(self, uf):
        # update fields in parent
        DPT.DPObject.append(self, uf)
        # update fields in child
        #

