import numpy as np
import glob
import os
import h5py
import PanGUI
import DataProcessingTools as DPT
import networkx as nx
from scipy.spatial.distance import cdist
# import time

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, suppress=True)

# dependencies
Args = {'RedoLevels': 0, 'SaveLevels': 0, 'Auto': 0, 'ArgsOnly': 0, 'ObjectLevel': 'Session', 'FileLineOfffset': 15,
        'DirName': 'RawData*', 'FileName': 'session*txt', 'TriggerVal1': 10, 'TriggerVal2': 20, 'TriggerVal3': 30,
        'MaxTimeDiff': 0.002}

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

G = nx.from_numpy_matrix(A)

# Vertices coordinates:
vertices = np.array([[-10, 10], [-5, 10], [0, 10], [5, 10], [10, 10], [-10, 5], [0, 5],
                     [10, 5], [-10, 0], [-5, 0], [0, 0], [5, 0], [10, 0], [-10, -5],
                     [0, -5], [10, -5], [-10, -10], [-5, -10], [0, -10], [5, -10], [10, -10]])
# Poster coordinates
poster_pos = np.array([[-5, -7.55], [-7.55, 5], [7.55, -5], [5, 7.55], [5, 2.45], [-5, -2.45]])
# Plot boundaries
xBound = [-12.5, 12.5, 12.5, -12.5, -12.5]
zBound = [12.5, 12.5, -12.5, -12.5, 12.5]
x1Bound = [-7.5, -2.5, -2.5, -7.5, -7.5]  # yellow pillar
z1Bound = [7.5, 7.5, 2.5, 2.5, 7.5]
x2Bound = [2.5, 7.5, 7.5, 2.5, 2.5]  # red pillar
z2Bound = [7.5, 7.5, 2.5, 2.5, 7.5]
x3Bound = [-7.5, -2.5, -2.5, -7.5, -7.5]  # blue pillar
z3Bound = [-2.5, -2.5, -7.5, -7.5, -2.5]
x4Bound = [2.5, 7.5, 7.5, 2.5, 2.5]  # green pillar
z4Bound = [-2.5, -2.5, -7.5, -7.5, -2.5]

FrameIntervalTriggers = np.array([1, 2], dtype=np.int)


class Unity:
    numSets = 0
    unityData = np.empty(0)
    unityTriggers = np.empty(0)
    unityTrialTime = np.empty(0)
    unityTime = np.empty(0)
    sumCost = np.empty(0)

    def __init__(self, num_sets, unity_data, unity_triggers, unity_trial_time, unity_time, sum_cost):
        self.numSets = num_sets
        self.unityData = unity_data
        self.unityTriggers = unity_triggers
        self.unityTrialTime = unity_trial_time
        self.unityTime = unity_time
        self.sumCost = sum_cost

    def info(self):
        print("numSets: ", self.numSets, "\n", "unityData: ", self.unityData.shape, "\n", "unityTriggers: ",
              self.unityTriggers.shape, "\n", "unityTrialTime: ", self.unityTrialTime.shape, "\n", "unityTime: ",
              self.unityTime.shape)

    def save(self):
        # save object to the current directory
        # Store to hdf5
        hf = h5py.File('unity.hdf5', 'w')
        hf.create_dataset('numSets', data=self.numSets)
        hf.create_dataset('unityData', data=self.unityData)
        hf.create_dataset('unityTriggers', data=self.unityTriggers)
        hf.create_dataset('unityTrialTime', data=self.unityTrialTime)
        hf.create_dataset('unityTime', data=self.unityTime)
        hf.create_dataset('sumCost', data=self.sumCost)
        hf.close()

    def plot(self):
        chose = input("Select which to plot (1.Trial 2.FrameIntervals 3.DurationDiffs 4.SumCost): ")
        if chose == '1':
            pp = PlotTrial(self)
            ppg = PanGUI.create_window(pp, indexer="trial")
        elif chose == '2':
            pp_in = PlotFrameIntervals(self)
            ppg = PanGUI.create_window(pp_in, indexer="trial")
        elif chose == '3':
            # Load data from rplparallel.hdf5
            data_rplparallel = h5py.File('rplparallel.hdf5', 'r')
            timeStamps = np.array(data_rplparallel.get('timeStamps'))
            data_rplparallel.close()

            pp_in = PlotDurationDiffs(self, timeStamps)
            ppg = PanGUI.create_window(pp_in, indexer="trial")
        else:
            pp_in = PlotSumCost(self)
            ppg = PanGUI.create_window(pp_in, indexer="trial")


def create():
    # empty unity data object
    unity = Unity(np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0),np.empty(0))
    # look for RawData_T * folder
    if bool(glob.glob("RawData*")):
        os.chdir(glob.glob("RawData*")[0])
        # look for session_1_*.txt in RawData_T*
        if bool(glob.glob("session*")):
            filename = glob.glob("session*")
            unity.numSets = len(filename)

            # load data of all session files into one matrix
            for index in range(0, len(filename)):
                if index == 0:
                    text_data = np.loadtxt(filename[index], skiprows=15)
                else:
                    text_data = np.concatenate((text_data, np.loadtxt(filename[index], skiprows=15)))

            # Move back to session directory from RawData directory
            os.chdir("..")

            # Unity Data
            # Calculate the displacement of each time stamp and its direction
            delta_x = np.diff(text_data[:, 2])
            delta_y = np.diff(text_data[:, 3])
            dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
            displacement_data = np.append(np.array([0]), dist)

            # The direction is in degrees, north set to 0, clockwise
            degree = np.degrees(np.arctan2(delta_y, delta_x))
            degree[degree < 0] = degree[degree < 0] + 360
            degree = degree - 90
            degree[degree < 0] = degree[degree < 0] + 360
            degree = 360 - degree
            degree[degree == 360] = 0
            direction_from_displacement = np.append(np.array([0]), degree)
            direction_from_displacement = np.where(displacement_data == 0, np.nan, direction_from_displacement)
            direction_and_direction = np.column_stack((direction_from_displacement, displacement_data))
            # Merge into the loaded text data to form Unity Data (matrix with 7 columns)
            unityData = np.append(text_data, direction_and_direction, axis=1)

            # Unity Triggers
            uT1 = np.where((text_data[:, 0] > 10) & (text_data[:, 0] < 20))
            uT2 = np.where((text_data[:, 0] > 20) & (text_data[:, 0] < 30))
            uT3 = np.where(text_data[:, 0] > 30)
            # Check if there is any incomplete trial
            utRows = [uT1[0].size, uT2[0].size, uT3[0].size]
            utMax = max(utRows)
            utMin = min(utRows)
            inCompleteTrials = utMax - utMin
            if inCompleteTrials != 0:
                print("Incomplete session! Last", inCompleteTrials, "trial discarded")
            unityTriggers = np.zeros((utMin, 3), dtype=int)
            unityTriggers[:, 0] = uT1[0][0:utMin]
            unityTriggers[:, 1] = uT2[0][0:utMin]
            unityTriggers[:, 2] = uT3[0]
            unityTriggers = unityTriggers.astype(int)

            # Unity Time
            unityTime = np.append(np.array([0]), np.cumsum(text_data[:, 1]))

            # Unity Trial Time
            totTrials = np.shape(unityTriggers)[0]
            unityTrialTime = np.empty((int(np.amax(uT3[0] - unityTriggers[:, 1])+2), totTrials))
            unityTrialTime.fill(np.nan)

            trial_counter = 0  # set up trial counter
            sumCost = np.zeros((404, 6))

            for a in range(0, totTrials):

                # Unity Trial Time
                uDidx = np.array(range(int(unityTriggers[a, 1] + 1), int(unityTriggers[a, 2] + 1)))
                numUnityFrames = uDidx.shape[0]
                tindices = np.array(range(0, numUnityFrames + 1))
                tempTrialTime = np.append(np.array([0]), np.cumsum(unityData[uDidx, 1]))
                unityTrialTime[tindices, a] = tempTrialTime

                # Sum Cost
                trial_counter = trial_counter + 1

                # get target identity
                target = unityData[unityTriggers[a, 2], 0] % 10

                # (starting position) get nearest neighbour vertex
                x = unityData[unityTriggers[a, 1], 2:4]
                s = cdist(vertices, x.reshape(1, -1))
                startPos = s.argmin()

                # (destination, target) get nearest neighbour vertex
                d = cdist(vertices, (poster_pos[int(target-1), :]).reshape(1, -1))
                destPos = d.argmin()

                idealCost, path = nx.bidirectional_dijkstra(G, destPos, startPos)

                mpath = np.empty(0)
                # get actual route taken(match to vertices)
                for b in range(0, (unityTriggers[a, 2]-unityTriggers[a, 1]+1)):
                    currPos = unityData[unityTriggers[a, 1] + b, 2:4]
                    # (current position)
                    cp = cdist(vertices, currPos.reshape(1, -1))
                    I3 = cp.argmin()
                    mpath = np.append(mpath, I3)

                pathdiff = np.diff(mpath)
                change = np.array([1])
                change = np.append(change, pathdiff)
                index = np.where(np.abs(change) > 0)
                actualRoute = mpath[index]
                actualCost = (actualRoute.shape[0]-1)*5
                # actualTime = index

                # Store summary
                sumCost[a, 0] = idealCost
                sumCost[a, 1] = actualCost
                sumCost[a, 2] = actualCost - idealCost
                sumCost[a, 3] = target
                sumCost[a, 4] = unityData[unityTriggers[a, 2], 0] - target

                if sumCost[a, 2] <= 0:  # least distance taken
                    sumCost[a, 5] = 1  # mark out trials completed via shortest route
                elif sumCost[a, 2] > 0 and sumCost[a, 4] == 30:
                    pathdiff = np.diff(actualRoute)

                    for c in range(0, pathdiff.shape[0]-1):
                        if pathdiff[c] == pathdiff[c + 1] * (-1):
                            timeingrid = np.where(mpath == actualRoute[c + 1])[0].shape[0]
                            if timeingrid > 165:
                                break
                            else:
                                sumCost[a, 5] = 1

            # Calculate performance
            errorInd = np.where(sumCost[:, 4] == 40)
            sumCost[errorInd, 5] = 0
            sumCost[errorInd[0] + 1, 5] = 0

            unity.sumCost = sumCost
            unity.unityData = unityData
            unity.unityTriggers = unityTriggers
            unity.unityTrialTime = unityTrialTime
            unity.unityTime = unityTime

            return unity
        else:
            return unity
    else:
        return unity


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)


# Class for the trial plot
class PlotTrial(DPT.objects.DPObject):
    def __init__(self, data, title="Test windwow", name="", ext="mat"):
        self.data = data
        self.title = title
        self.dirs = [""]
        self.setidx = np.zeros(data.unityTriggers.shape[0], dtype=np.int)

    def load(self):
        fname = os.path.join(self.name, self.ext)
        if os.path.isfile(fname):
            if self.ext == "mat":
                dd = mio.loadmat(fname, squeeze_me=True)

    def update_idx(self, i):
        return max(0, min(i, self.data.shape[0] - 1))

    def plot(self, i, ax=None, overlay=False):
        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()
        ax.plot(xBound, zBound, color='k', linewidth=1.5)
        ax.plot(x1Bound, z1Bound, 'k', LineWidth=1)
        ax.plot(x2Bound, z2Bound, 'k', LineWidth=1)
        ax.plot(x3Bound, z3Bound, 'k', LineWidth=1)
        ax.plot(x4Bound, z4Bound, 'k', LineWidth=1)
        ax.plot(self.data.unityData[int(self.data.unityTriggers[i, 1]): int(self.data.unityTriggers[i, 2]), 2],
                self.data.unityData[int(self.data.unityTriggers[i, 1]): int(self.data.unityTriggers[i, 2]), 3],
                'b+', LineWidth=1)

        # plot end point identifier
        ax.plot(self.data.unityData[self.data.unityTriggers[i, 2], 2],
                self.data.unityData[self.data.unityTriggers[i, 2], 3], 'k.', MarkerSize=10)
        return ax


# Class for the trial FrameIntervals plot
class PlotFrameIntervals(DPT.objects.DPObject):
    def __init__(self, data, title="Test windwow", name="", ext="mat"):
        self.data = data
        self.title = title
        self.dirs = [""]
        self.setidx = np.zeros(data.unityTriggers.shape[0], dtype=np.int)

    def load(self):
        fname = os.path.join(self.name, self.ext)
        if os.path.isfile(fname):
            if self.ext == "mat":
                dd = mio.loadmat(fname, squeeze_me=True)

    def update_idx(self, i):
        return max(0, min(i, self.data.shape[0] - 1))

    def plot(self, i, ax=None, overlay=False):
        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()

        indices = self.data.unityTriggers[i, FrameIntervalTriggers]
        uData = self.data.unityData[(indices[0] + 1):(indices[1]+1), 1]
        ax.stem(uData, basefmt=" ", use_line_collection=True)
        ax.set_xlabel('Frames')
        ax.set_ylabel('Interval (s)')

        return ax


# Class for the DurationDiff plot
class PlotDurationDiffs(DPT.objects.DPObject):
    def __init__(self, data, timeStamps, title="Test windwow", name="", ext="mat"):
        self.data = data
        self.timeStamps = timeStamps
        self.title = title
        self.dirs = [""]
        self.setidx = np.zeros(1, dtype=np.int)

    def load(self):
        fname = os.path.join(self.name, self.ext)
        if os.path.isfile(fname):
            if self.ext == "mat":
                dd = mio.loadmat(fname, squeeze_me=True)

    def update_idx(self, i):
        return max(0, min(i, self.data.shape[0] - 1))

    def plot(self, i, ax=None, overlay=False):
        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()

        totTrials = self.data.unityTriggers.shape[0]
        setIndex = np.array([0, totTrials])
        uTrigs = self.data.unityTriggers
        uTime = self.data.unityTime
        # add 1 to start index since we want the duration between triggers
        startind = uTrigs[:, 0]+1
        endind = uTrigs[:, 2]+1
        starttime = uTime[startind]
        endtime = uTime[endind]
        trialDurations = endtime - starttime
        # load the rplparallel object to get the Ripple timestamps
        rpTrialDur = self.timeStamps[:, 2] - self.timeStamps[:, 0]
        # multiply by 1000 to convert to ms
        duration_diff = (trialDurations - rpTrialDur)*1000
        num_bin = (np.amax(duration_diff) - np.amin(duration_diff))/200
        ax.hist(x=duration_diff, bins=int(num_bin))
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency')
        ax.set_yscale("log")
        ax.grid(axis="y")

        return ax


# Class for the DurationDiff plot
class PlotSumCost(DPT.objects.DPObject):
    def __init__(self, data, title="Test windwow", name="", ext="mat"):
        self.data = data
        self.title = title
        self.dirs = [""]
        self.setidx = np.zeros(1, dtype=np.int)

    def load(self):
        fname = os.path.join(self.name, self.ext)
        if os.path.isfile(fname):
            if self.ext == "mat":
                dd = mio.loadmat(fname, squeeze_me=True)

    def update_idx(self, i):
        return max(0, min(i, self.data.shape[0] - 1))

    def plot(self, i, ax=None, overlay=False):
        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()

        totTrials = self.data.unityTriggers.shape[0]
        xind = np.arange(0, totTrials)
        # Calculate optimal width
        width = np.min(np.diff(xind)) / 3
        ax.bar(xind-width/2, self.data.sumCost[xind, 0], width, color='yellow')
        ax.bar(xind+width/2, self.data.sumCost[xind, 1], width, color='cyan')
        ax1 = ax.twinx()
        ratio = np.divide(self.data.sumCost[xind, 1], self.data.sumCost[xind, 0])
        markerline, stemlines, baseline = ax1.stem(xind, ratio, 'magenta', markerfmt='mo', basefmt=" ", use_line_collection=True, label='Ratio')
        markerline.set_markersize(2)
        stemlines.set_linewidth(0.4)
        markerline.set_markerfacecolor('none')
        align_yaxis(ax,0,ax1,0)
        # ax.grid(axis="y")

        return ax


def load():
    # Load data from unity.hdf5
    data_unity = h5py.File('unity.hdf5', 'r')
    n1 = np.array(data_unity.get('numSets'))
    n2 = np.array(data_unity.get('unityData'))
    n3 = np.array(data_unity.get('unityTriggers'))
    n4 = np.array(data_unity.get('unityTrialTime'))
    n5 = np.array(data_unity.get('unityTime'))
    n6 = np.array(data_unity.get('sumCost'))
    unity_data_load = Unity(n1, n2, n3, n4, n5, n6)
    data_unity.close()

    return unity_data_load

# tic = time.perf_counter()
# b = create()
# toc = time.perf_counter()
# print(f"{toc - tic} seconds")


