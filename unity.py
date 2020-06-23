import numpy as np
import glob
import os
import h5py
import matplotlib.pyplot as plt
import PanGUI
import DataProcessingTools as DPT
import matplotlib.patches as patches

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, suppress=True)

# dependencies
Args = {'RedoLevels': 0, 'SaveLevels': 0, 'Auto': 0, 'ArgsOnly': 0, 'ObjectLevel': 'Session', 'FileLineOfffset': 15,
        'DirName': 'RawData*', 'FileName': 'session*txt', 'TriggerVal1': 10, 'TriggerVal2': 20, 'TriggerVal3': 30,
        'MaxTimeDiff': 0.002}


class Data:
    def __init__(self):
        self.numSets = 0
        self.unityData = np.empty(0)
        self.unityTriggers = np.empty(0)
        self.unityTrialTime = np.empty(0)
        self.unityTime = np.empty(0)

    def info(self):
        print("numSets: ", self.numSets, "\n", "unityData: ", self.unityData.shape, "\n", "unityTriggers: ",
              self.unityTriggers.shape, "\n", "unityTrialTime: ", self.unityTrialTime.shape, "\n", "unityTime: ",
              self.unityTime.shape)


def create_object():
    # empty unity data object
    unity = Data()
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
            unityTriggers[:, 0] = uT1[0][0:utMin].astype(int)
            unityTriggers[:, 1] = uT2[0][0:utMin].astype(int)
            unityTriggers[:, 2] = uT3[0].astype(int)
            unityTriggers = unityTriggers.astype(int)

            # Unity Trial Time
            totTrials = np.shape(unityTriggers)[0]
            unityTrialTime = np.empty((int(np.amax(uT3[0] - unityTriggers[:, 1])+2), totTrials))
            unityTrialTime.fill(np.nan)

            for a in range(0, totTrials):
                uDidx = np.array(range(int(unityTriggers[a, 1]+1), int(unityTriggers[a, 2]+1)))
                numUnityFrames = uDidx.shape[0]
                tindices = np.array(range(0, numUnityFrames+1))
                tempTrialTime = np.append(np.array([0]), np.cumsum(unityData[uDidx, 1]))
                unityTrialTime[tindices, a] = tempTrialTime

            # Unity Time
            unityTime = np.append(np.array([0]), np.cumsum(text_data[:, 1]))

            unity.unityData = unityData
            unity.unityTriggers = unityTriggers
            unity.unityTrialTime = unityTrialTime
            unity.unityTime = unityTime

            return unity
        else:
            return unity
    else:
        return unity


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
        ax.plot(self.data.unityData[int(self.data.unityTriggers[i, 1]): int(self.data.unityTriggers[i, 2] - 1), 2],
                self.data.unityData[int(self.data.unityTriggers[i, 1]): int(self.data.unityTriggers[i, 2] - 1), 3],
                'b+', LineWidth=1)

        # plot end point identifier
        ax.plot(self.data.unityData[self.data.unityTriggers[i, 2], 2],
                self.data.unityData[self.data.unityTriggers[i, 2], 3], 'k.', MarkerSize=20)
        return ax


def plot(unity):

    pp = PlotTrial(unity)
    ppg = PanGUI.create_window(pp, indexer="trial")


def main():
    unity_data_generated = create_object()
    plot(unity_data_generated)
    # unity_data_generated.info()

    # save object to the current directory
    # data dictionary
    # data = {"numSets": unity_data_generated.numSets, "unityData": unity_data_generated.unityData,
    #         "unityTriggers": unity_data_generated.unityTriggers, "unityTrialTime": unity_data_generated.unityTrialTime,
    #         "unityTime": unity_data_generated.unityTime}

    # hdf5
    # hf = h5py.File('test.h5', 'w')
    # hf.create_dataset('numSets', data=unity_data_generated.numSets)
    # hf.create_dataset('unityData', data=unity_data_generated.unityData)
    # hf.create_dataset('unityTriggers', data=unity_data_generated.unityTriggers)
    # hf.create_dataset('unityTrialTime', data=unity_data_generated.unityTrialTime)
    # hf.create_dataset('unityTime', data=unity_data_generated.unityTime)
    # hf.close()


if __name__ == "__main__":
    main()


