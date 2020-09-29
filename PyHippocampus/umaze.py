import DataProcessingTools as DPT
from pylab import gcf, gca
import numpy as np
import os
import glob
from .unity import Unity
import scipy
import networkx as nx
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Umaze(DPT.DPObject):
    filename = "umaze.hkl"
    argsList = [('GridSteps', 40), ('OverallGridSize', 25), ('MinObs', 5)]
    level = "session"

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):
        # load the unity object to get the data
        uf = Unity()

        # check if the Unity object is empty
        if len(uf.dirs) == 0:
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)
        else:
            print('Creating Umaze')
            DPT.DPObject.create(self, *args, **kwargs)
            unityData = uf.unityData[0]
            unityTriggers = uf.unityTriggers[0]
            unityTrialTime = uf.unityTrialTime[0]
            unityTime = uf.unityTime[0]
            sumCost = uf.sumCost[0]
            totTrials = np.shape(unityTriggers)[0]
            GridSteps = self.args['GridSteps']
            OverallGridSize = self.args['OverallGridSize']
            gridBins = GridSteps * GridSteps
            oGS2 = OverallGridSize/2
            gridSize = OverallGridSize/GridSteps
            gpEdges = np.arange(0, (gridBins+1))
            horGridBound = np.arange(-oGS2, oGS2+gridSize, gridSize)
            vertGridBound = horGridBound
            ret = stats.binned_statistic_2d(unityData[:, 2], unityData[:, 3], np.zeros(
                (np.shape(unityData[:, 2]))), bins=(horGridBound, vertGridBound), expand_binnumbers=True)
            binH = ret.binnumber[0]
            binV = ret.binnumber[1]
            gridPosition = binH + ((binV - 1) * GridSteps)
            gpDurations = np.zeros((gridBins, totTrials))
    
            # line 151-354
            trialCounter = 0
            gpreseti = 0
            sessionTime = np.zeros((np.shape(gridPosition)[0], 3))
            sTi = 1
            count = 0
            for a in range(totTrials):
                trialCounter = trialCounter + 1
                uDidx = np.arange(
                    int(unityTriggers[a, 1]) + 1, int(unityTriggers[a, 2]) + 1)
                numUnityFrames = np.shape(uDidx)[0]
                tindices = np.arange(0, (numUnityFrames+1))
                arr = unityData[uDidx[0]:uDidx[numUnityFrames-1], 1]
                tempTrialTime = np.array([0, np.cumsum(arr)])
                tstart = unityTime[uDidx[0]]
                tend = unityTime[uDidx[-1]+1]
                # get grid positions for this trial
                tgp = []
                for i in range(numUnityFrames):
                    tgp.append(gridPosition[uDidx[i]])
                tgp = np.array(tgp)
    
                if tempTrialTime[1][np.shape(tempTrialTime[1])[0]-1] - tempTrialTime[0] != 0:
                    sessionTime[sTi, 0] = np.array([tstart])
                    sessionTime[sTi, 1] = np.array([tgp[0]])
                    sessionTime[sTi, 2] = np.array([0])
                    sTi += 1
                    gpc = np.where(np.diff(tgp) != 0)
    
                    ngpc = np.shape(gpc)[1]
                    temp_0 = []
                    temp_1 = []
                    for i in gpc[0][:]:
                        temp_0.append(unityTrialTime[i+1, a])
                        temp_1.append(tgp[i+1])
                    temp_0 += tstart
                    temp_0 = np.array(temp_0)
                    temp_1 = np.array(temp_1)
    
                    sessionTime[sTi:(sTi+ngpc), 0] = temp_0
                    sessionTime[sTi:(sTi+ngpc), 1] = temp_1
    
                    sTi += ngpc
                    count += np.shape(gpc)[1]
                    if np.shape(gpc)[1] != 0:
                        if gpc and (gpc[0][ngpc-1] == (numUnityFrames-1)):
                            sTi -= 1
                else:
                    sessionTime[sTi, 0] = tstart
                    sTi += 1
    
                sessionTime[sTi, 0] = np.array([tend])
                sessionTime[sTi, 1] = np.array([0])
                sTi += 1
                utgp = np.unique(tgp)
                for pidx in range(np.shape(utgp)[0]):
                    tempgp = utgp[pidx]
                    utgpidx = np.where(tgp == tempgp)
                    utgpidx = uDidx[utgpidx]
                    gpDurations[tempgp-1, a] = np.sum(unityData[utgpidx+1, 1])
                gridPosition[gpreseti:uDidx[0]] = 0
                gpreseti = unityTriggers[a, 2]+1
    
            snum = sTi - 1
            sTime = sessionTime[0:snum+1, :]
            sTime[0:snum, 2] = np.diff(sTime[:, 0])
            sTP = np.sort(sTime[:, 1], axis=0)
            sTPi = np.argsort(sTime[:, 1], axis=0)
            sTPsi = np.where(np.diff(sTP) != 0)
            temp = sTPsi[0]
            temp += 1
            sTPsi = np.array(temp)
    
            if sTP[0] == -1:
                sTPsi = sTPsi[1:]
            temp = []
            for i in range(1, np.shape(sTPsi)[0]):
                temp.append(sTPsi[i]-1)
    
            temp.append(np.shape(sTP)[0]-1)
            temp = np.array(temp)
    
            sTPind = np.concatenate((sTPsi, temp))
            sTPind = np.reshape(sTPind, (int(np.shape(sTPind)[0]/2), 2), order='F')
    
            sTPin = np.diff(sTPind, 1, 1)
            temp = []
            for i in range(np.shape(sTPin)[0]):
                temp.append(sTPin[i][0])
            sTPin = np.array(temp)
    
            sortedGPindinfo = np.concatenate((sTP[sTPsi], sTPind[:, 0]))
            sortedGPindinfo = np.concatenate((sortedGPindinfo, sTPind[:, 1]))
            sortedGPindinfo = np.concatenate((sortedGPindinfo, sTPin))
            sortedGPindinfo = np.reshape(sortedGPindinfo, (int(
                np.shape(sortedGPindinfo)[0]/4), 4), order='F')
    
            gp2ind = np.nonzero(
                np.in1d(np.arange(0, gridBins), sortedGPindinfo[:, 0]))[0]
            sTPinm = np.where(sTPin > (self.args['MinObs']-2))
    
            sTPsi2 = sTPsi[sTPinm]
            sTPin2 = sTPin[sTPinm]
            sTPu = sTP[sTPsi2]
            nsTPu = np.shape(sTPu)[0]
            sTPind2 = sTPind[sTPinm, :]
            ou_i = np.zeros((nsTPu, 1))
    
            for pi in range(nsTPu):
                ou_i[pi] = np.sum(
                    sTime[sTPi[sTPind2[0][pi, 0]:sTPind2[0][pi, 1]], 2])
    
            self.GridSteps = GridSteps
            self.OverallGridSize = OverallGridSize
            self.oGS2 = oGS2
            self.gridSize = gridSize
            self.horGridBound = horGridBound
            self.vertGridBound = vertGridBound
            self.gpEdges = gpEdges
            self.gridPosition = gridPosition
            self.gpDurations = gpDurations
            self.setIndex = np.array([[0], [totTrials]])
            self.processTrials = np.where(sumCost[:, 5] == 1)
            self.sessionTime = sTime
            self.sortedGPindices = sTPi
            self.sortedGPindinfo = sortedGPindinfo
            self.sGPi_minobs = sTPinm
            self.sTPu = sTPu
            self.nsTPu = nsTPu
            self.ou_i = ou_i
            self.P_i = ou_i / np.sum(ou_i)
            self.gp2ind = gp2ind
            self.unityTriggers = uf.unityTriggers
            self.unityTrialTime = uf.unityTrialTime
            self.unityData = unityData
            self.unityTime = unityTime
    
            # plot heatmap
            arr = []
            for i in range(gridBins+1):
                lst = []
                for j in range(np.shape(self.sessionTime)[0]):
                    if self.sessionTime[j, 1] == i:
                        lst.append(self.sessionTime[j, 2])
                lst = np.array(lst)
                arr.append(lst)
            arr = np.array(arr)
            for k in range(np.shape(arr)[0]):
                arr[k] = np.cumsum(arr[k])
            lst = []
            for i in range(1, gridBins+1):
                temp = []
                if arr[i].size == 0:
                    temp.append(0)
                else:
                    temp.append(arr[i][np.shape(arr[i])[0]-1])
                #temp = np.array(temp)
                lst.append(temp)
            lst = np.array(lst)
            matrix = np.reshape(lst, (self.GridSteps, self.GridSteps), order='C')
            matrix_r = []
            for i in reversed(range(np.shape(matrix)[0])):
                matrix_r.append(matrix[i])
            matrix_r = np.array(matrix_r)
            df = pd.DataFrame(matrix_r)
            self.matrix = df 
        
    def plot(self, i = None, ax = None, getNumEvents = False, getLevels = False, getPlotOpts = False, overlay = False, **kwargs):
        # set plot options
        plotopts = {}
        if getPlotOpts:
            return plotopts

        # Extract the recognized plot options from kwargs
        for (k, v) in plotopts.items():
            plotopts[k] = kwargs.get(k, v)

        if getLevels:
            return ['session']

        if getNumEvents:
            return 1, 0 

        if ax is None:
            ax = gca()
        
        if not overlay:
            ax.clear()

        im = ax.pcolormesh(self.matrix)
        direct = os.getcwd()
        day = DPT.levels.get_shortname('day', direct)
        session = DPT.levels.get_shortname("session", direct)
        title = 'D' + day + session
        ax.set_title(title)
        # Uncomment colorbar line after PanGUI has been fixed. 
        # plt.colorbar(im, ax = ax)
        return ax

    def append(self, uf):
        # update fields in parent
        DPT.DPObject.append(self, uf)
        # update fields in child
