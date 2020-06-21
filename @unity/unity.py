import numpy as np
import glob
import os
# import pickle

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
            unityTriggers = np.zeros((utMin, 3))
            unityTriggers[:, 0] = uT1[0][0:utMin]
            unityTriggers[:, 1] = uT2[0][0:utMin]
            unityTriggers[:, 2] = uT3[0]

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


def main():
    unity_data_generated = create_object()
    unity_data_generated.info()

    # save object to the current directory

    # with open('unity.pkl', 'wb') as output:
    #     pickle.dump(unity_data_generated, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()


