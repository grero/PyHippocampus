import DataProcessingTools as DPT
from pylab import gcf, gca
import numpy as np
import os
import glob
from . import unity
import scipy
import networkx as nx


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
        totTrials = np.shape(unityTriggers)[0]
        A = np.array([[0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 5, 0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 0, 0, 5, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0, 5, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0]])        
        gridSteps = args.GridSteps
        gridBins = gridSteps * gridSteps
        oGS2 = overallGridSize/2
        gridSize = overallGridSize/gridSteps
        horGridBound = np.arange(-oGS2, oGS2, gridSize)
        vertGridBound = horGridBound
        gridPosition, binH, binV = np.histogram2d(unityData[:, 2],unityData[:, 3],bins = (horGridBound, vertGridBound))
        gridPosition = binH + ((binV - 1) * gridSteps)  
        vertices = np.array([[-10, 10], [-5, 10], [0, 10], [5, 10], [10, 10], [-10, 5], [0, 5],
                             [10, 5], [-10, 0], [-5, 0], [0, 0], [5, 0], [10, 0], [-10, -5],
                             [0, -5], [10, -5], [-10, -10], [-5, -10], [0, -10], [5, -10], [10, -10]])
        posterpos = np.array([[-5, -7.55], [-7.55, 5], [7.55, -5], [5, 7.55], [5, 2.45], [-5, -2.45]])
        
        #line 151-354
        trialCounter = 0
        gpreseti = 0
        sessionTime = np.zeros((np.shape(gridPosition)[0], 3))
        
        sTi = 1
        for a in range(totTrials):
            trialCounter = trialCounter + 1
            # indeces minus 1 in python
            target = unityData[unityTriggers[a,2],0] % 10
            S = scipy.spatial.distance.cdist(vertices, unityData[unityTriggers[a,1],2:4], 'euclidean')
            M1 = np.amin(S)
            I1 = np.argmin(S)
            startPos = I1
            D = scipy.spatial.distance.cdist(vertices, posterpos[target-1,:], 'euclidean')
            M2 = np.amin(D)
            I2 = np.argmin(D)
            destPos = I2      
            idealCost, idealroute = nx.bidirectional_dijkstra(A, startPos, destPos)
            
            mpath = np.empty(0)
            # get actual route taken(match to vertices)
            for b in range(0, (unityTriggers[a, 2] - unityTriggers[a, 1] + 1)):
                curr_pos = unityData[unityTriggers[a, 1] + b, 2:4]
                # (current position)
                cp = cdist(vertices, curr_pos.reshape(1, -1))
                I3 = cp.argmin()
                mpath = np.append(mpath, I3)

            path_diff = np.diff(mpath)
            change = np.array([1])
            change = np.append(change, path_diff)
            index = np.where(np.abs(change) > 0)
            actual_route = mpath[index]
            actual_cost = (actual_route.shape[0] - 1) * 5
            actualTime = np.array(index) 
            actualTime = np.append(np.array(np.shape(mpath)[0]))
            actualTime = np.diff(actualTime)+1            

            # Store summary
            sumCost[a, 0] = ideal_cost
            sumCost[a, 1] = actual_cost
            sumCost[a, 2] = actual_cost - ideal_cost
            sumCost[a, 3] = target
            sumCost[a, 4] = unityData[unityTriggers[a, 2], 0] - target
            #sumRoute(a,1:size(idealroute,2)) = idealroute
            #sumActualRoute(a,1:size(actualRoute,1)) = actualRoute
            #sumActualTime(a,1:size(actualTime,1)) = actualTime            
            
            uDidx = np.array(range(int(unityTriggers[a, 1] + 1), int(unityTriggers[a, 2] + 1)))
            numUnityFrames = np.shape(uDidx)[1]
            tempTrialTime = np.array([0], [np.cumsum(unityData[uDidx,1])])
            tstart = unityTime[uDidx[0, 0]]
            tend = unityTime[uDidx[0, np.shape(uDix)[1]-1]]
            
            # get grid positions for this trial
            tgp = gridPosition[uDidx]
            binHt = binH[uDidx]
            binVt = binV[uDidx]           
            if tempTrialTime[np.shape(tempTrialTime)[0]-1] - tempTrialTime[0] != 0:
                sessionTime[sTi, 0] = np.array([tstart])
                sessionTime[sTi, 1] = np.array([tgp[0]])
                sessionTime[sTi, 2] = np.array([0])
                sTi += 1
                gpc = np.where(np.diff[tgp] != 0)
                ngpc = np.shape(gpc)[0]
                sessionTime[sTi:(sTi+ngpc-1), 0] = np.array([unityTrialTime[gpc+2, a]+tstart])
                sessionTime[sTi:(sTi+ngpc-1), 1] = np.array([tgp[gpc+1]])
                sTi += ngpc
                if (gpc.size != 0) and (gpc[np.shape(gpc)[0]-1] == (numUnityFrames-1)):
                    sTi -= 1
            else:
                sessionTime[sTi, 0] = tstart
                sTi += 1
            
            sessionTimesTi[sTi, 0] = np.array([tend])
            sessionTimesTi[sTi, 1] = np.array([0])
            sTi += 1
            utgp = np.unique(tgp)
            
            for pidx in range(np.shape(utgp)[0]):
                tempgp = utgp[pidx]
                if tempgp == 58:
                    print('test')
                utgpidx = np.where(tgp == tempgp)
                utgpidx = uDidx[utgpidx]
                gpDurations[tempgp, a] = np.sum(unityData[utgpidx+1, 1])                    
                                          
            gridPosition[gpreseti:uDidx[0]] = 0
            gpreseti = unityTriggers(a,2)+1               
        
        print('speed thresholding portion')
        
        
        snum = sTi - 1
        sTime = sessionTime[0:snum,:]
        sTime[1:(snum-1),2] = np.diff(sTime[:,0])
        sTP, sTPi = np.sort(sTime[:,1])
        sTPsi = np.where(np.diff(sTP) != 0) + 1
        if sTP[0] == -1:
            sTPsi = sTPsi[1:]  
        sTPind = np.array(sTPsi[[sTPsi[1:]-1], [np.shape(sTP)[0]]])
        sTPin = np.diff(sTPind,1,1) + 1
        sortedGPindinfo = np.array([sTP(sTPsi)] [sTPind] [sTPin])
        _, gp2ind = np.in1d(np.arange(0,gridBins), sortedGPindinfo[:,0])
        sTPinm = np.where(sTPin>(self.arges['MinObs']-1))
        
        sTPsi2 = sTPsi[sTPinm]
        sTPin2 = sTPin[sTPinm]
        sTPu = sTP[sTPsi2]
        nsTPu = np.shape(sTPu)[0]
        sTPind2 = sTPind[sTPinm,:]
        ou_i = np.zeros(nsTPu,1)
        
        for pi in range(nsTPu):
            ou_i[pi] = np.sum(sTime[sTPi[sTPind2[pi,0]:sTPind2[pi,1]],3])
        
        
        
                

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
    

