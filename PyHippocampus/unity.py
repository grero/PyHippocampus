import PanGUI
import DataProcessingTools as DPT
from pylab import gcf, gca
import numpy as np
import os
import glob
import networkx as nx
from scipy.spatial.distance import cdist
from . import rplparallel
# import rplparallel

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=4, suppress=True)

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


class Unity(DPT.DPObject):
    filename = "unity.hkl"
    argsList = [("FileLineOffset", 15), ("DirName", 'RawData*'), ("FileName", 'session*'), ('TriggerVal1', 10),
                ('TriggerVal2', 20), ('TriggerVal3', 30)]

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, normpath=False, *args, **kwargs)

    def create(self, *args, **kwargs):
        # set plot options
        self.plotopts = {"Plot Option": DPT.objects.ExclusiveOptions(["Trial", "FrameIntervals", "DurationDiffs",
                                                                      "SumCost"], 0),
                         "FrameIntervalTriggers": {"from": 1.0, "to": 2.0}}
        self.indexer = self.getindex("trial")

        # initialization
        self.numSets = 0
        self.sumCost = np.empty(0)
        self.unityData = np.empty(0)
        self.unityTriggers = np.empty(0)
        self.unityTrialTime = np.empty(0)
        self.unityTime = np.empty(0)

        # look for RawData_T * folder
        if bool(glob.glob(self.args["DirName"])):
            os.chdir(glob.glob(self.args["DirName"])[0])
            # look for session_1_*.txt in RawData_T*
            if bool(glob.glob(self.args["FileName"])):
                filename = glob.glob(self.args["FileName"])
                self.numSets = len(filename)

                # load data of all session files into one matrix
                for index in range(0, len(filename)):
                    if index == 0:
                        text_data = np.loadtxt(filename[index], skiprows=self.args["FileLineOffset"])
                    else:
                        text_data = np.concatenate((text_data, np.loadtxt(filename[index], skiprows=self.args["FileLineOffset"])))

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
                uT1 = np.where((text_data[:, 0] > self.args["TriggerVal1"]) & (text_data[:, 0] < self.args["TriggerVal2"]))
                uT2 = np.where((text_data[:, 0] > self.args["TriggerVal2"]) & (text_data[:, 0] < self.args["TriggerVal3"]))
                uT3 = np.where(text_data[:, 0] > self.args["TriggerVal3"])
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
                unityTrialTime = np.empty((int(np.amax(uT3[0] - unityTriggers[:, 1]) + 2), totTrials))
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
                    start_pos = s.argmin()

                    # (destination, target) get nearest neighbour vertex
                    d = cdist(vertices, (poster_pos[int(target - 1), :]).reshape(1, -1))
                    des_pos = d.argmin()

                    ideal_cost, path = nx.bidirectional_dijkstra(G, des_pos, start_pos)

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
                    # actualTime = index

                    # Store summary
                    sumCost[a, 0] = ideal_cost
                    sumCost[a, 1] = actual_cost
                    sumCost[a, 2] = actual_cost - ideal_cost
                    sumCost[a, 3] = target
                    sumCost[a, 4] = unityData[unityTriggers[a, 2], 0] - target

                    if sumCost[a, 2] <= 0:  # least distance taken
                        sumCost[a, 5] = 1  # mark out trials completed via shortest route
                    elif sumCost[a, 2] > 0 and sumCost[a, 4] == 30:
                        path_diff = np.diff(actual_route)

                        for c in range(0, path_diff.shape[0] - 1):
                            if path_diff[c] == path_diff[c + 1] * (-1):
                                timeingrid = np.where(mpath == actual_route[c + 1])[0].shape[0]
                                if timeingrid > 165:
                                    break
                                else:
                                    sumCost[a, 5] = 1

                # Calculate performance
                error_ind = np.where(sumCost[:, 4] == 40)
                sumCost[error_ind, 5] = 0
                sumCost[error_ind[0] + 1, 5] = 0

                self.sumCost = sumCost
                self.unityData = unityData
                self.unityTriggers = unityTriggers
                self.unityTrialTime = unityTrialTime
                self.unityTime = unityTime
                self.setidx = np.zeros((self.unityTriggers.shape[0],), dtype=np.int)

        # check if we need to save the object, with the default being 0
        if kwargs.get("saveLevel", 0) > 0:
            self.save()

    def update_idx(self, i):
        return max(0, min(i, self.unityTriggers.shape[0]-1))

    def plot(self, i, ax=None, overlay=False):
        self.current_idx = i
        # fig = ax.get_figure()
        # ax_list = fig.get_axes()
        # for ax in ax_list:
        #     ax.clear()
        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()
        plot_type = self.plotopts["Plot Option"].selected()
        if plot_type == "Trial":

            ax.plot(xBound, zBound, color='k', linewidth=1.5)
            ax.plot(x1Bound, z1Bound, 'k', LineWidth=1)
            ax.plot(x2Bound, z2Bound, 'k', LineWidth=1)
            ax.plot(x3Bound, z3Bound, 'k', LineWidth=1)
            ax.plot(x4Bound, z4Bound, 'k', LineWidth=1)
            ax.plot(self.unityData[int(self.unityTriggers[i, 1]): int(self.unityTriggers[i, 2]), 2],
                    self.unityData[int(self.unityTriggers[i, 1]): int(self.unityTriggers[i, 2]), 3],
                    'b+', LineWidth=1)

            # plot end point identifier
            ax.plot(self.unityData[self.unityTriggers[i, 2], 2],
                    self.unityData[self.unityTriggers[i, 2], 3], 'k.', MarkerSize=10)
            route_str = str(self.sumCost[i, 1])
            short_str = str(self.sumCost[i, 0])
            ratio_str = str(self.sumCost[i, 1] / self.sumCost[i, 0])
            title = ' T: ' + str(i) + ' Route: ' + route_str + ' Shortest: ' + short_str + ' Ratio: ' + ratio_str

            dir_name = self.dirs[0]
            subject = DPT.levels.get_shortname("subject", dir_name)
            date = DPT.levels.get_shortname("day", dir_name)
            session = DPT.levels.get_shortname("session", dir_name)
            title = subject + date + session + title
            ax.set_title(title)

        elif plot_type == "FrameIntervals":

            rl = rplparallel.RPLParallel()
            time_stamps = rl.timeStamps
            frame_interval_triggers = np.array([self.plotopts["FrameIntervalTriggers"]["from"],
                                                self.plotopts["FrameIntervalTriggers"]["to"]], dtype=np.int)
            indices = self.unityTriggers[i, frame_interval_triggers]
            u_data = self.unityData[(indices[0] + 1):(indices[1] + 1), 1]
            markerline, stemlines, baseline = ax.stem(u_data, basefmt=" ", use_line_collection=True)
            stemlines.set_linewidth(0.5)
            markerline.set_markerfacecolor('none')

            ax.set_ylim(bottom=0)
            ax.set_xlabel('Frames')
            ax.set_ylabel('Interval (s)')
            start = time_stamps[i, 1]
            end = time_stamps[i, 2]
            rp_trial_dur = end - start
            uet = np.cumsum(u_data)
            title = " Trial " + str(i) + ' Duration disparity: ' + str(1000 * (uet[-1] - rp_trial_dur)) + ' ms'

            dir_name = self.dirs[0]
            subject = DPT.levels.get_shortname("subject", dir_name)
            date = DPT.levels.get_shortname("day", dir_name)
            session = DPT.levels.get_shortname("session", dir_name)
            title = subject + date + session + title
            ax.set_title(title)

        elif plot_type == "DurationDiffs":

            # load the rplparallel object to get the Ripple timestamps
            rl = rplparallel.RPLParallel()
            time_stamps = rl.timeStamps
            u_triggers = self.unityTriggers
            u_time = self.unityTime

            # add 1 to start index since we want the duration between triggers
            start_ind = u_triggers[:, 0] + 1
            end_ind = u_triggers[:, 2] + 1
            start_time = u_time[start_ind]
            end_time = u_time[end_ind]
            trial_durations = end_time - start_time

            rp_trial_dur = time_stamps[:, 2] - time_stamps[:, 0]
            # multiply by 1000 to convert to ms
            duration_diff = (trial_durations - rp_trial_dur) * 1000
            num_bin = (np.amax(duration_diff) - np.amin(duration_diff)) / 200

            ax.hist(x=duration_diff, bins=int(num_bin))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency')
            ax.set_yscale("log")
            ax.grid(axis="y")

            dir_name = self.dirs[0]
            subject = DPT.levels.get_shortname("subject", dir_name)
            date = DPT.levels.get_shortname("day", dir_name)
            session = DPT.levels.get_shortname("session", dir_name)
            ax.set_title('Unity trial duration - Ripple trial duration ' + subject + date + session)

        elif plot_type == "SumCost":
            tot_trials = self.unityTriggers.shape[0]
            xind = np.arange(0, tot_trials)
            # Calculate optimal width
            width = np.min(np.diff(xind)) / 3
            ax.bar(xind - width / 2, self.sumCost[xind, 0], width, color='yellow', label="Shortest")
            ax.bar(xind + width / 2, self.sumCost[xind, 1], width, color='cyan', label="Route")
            ax1 = ax.twinx()
            ratio = np.divide(self.sumCost[xind, 1], self.sumCost[xind, 0])
            markerline, stemlines, baseline = ax1.stem(xind, ratio, 'magenta', markerfmt='mo', basefmt=" ",
                                                       use_line_collection=True, label='Ratio')
            # markerline.set_markersize(5)
            stemlines.set_linewidth(0.4)
            markerline.set_markerfacecolor('none')
            ax1.set_ylim(bottom=0)
            ax1.grid(axis="y")
            ax1.spines['right'].set_color('magenta')
            ax1.tick_params(axis='y', colors='magenta')
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax1.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper right")

            dir_name = self.dirs[0]
            subject = DPT.levels.get_shortname("subject", dir_name)
            date = DPT.levels.get_shortname("day", dir_name)
            session = DPT.levels.get_shortname("session", dir_name)
            ax.set_title(subject + date + session)

        return ax


# pg = Unity(saveLevel=1)
# print(pg.get_filename())
# ag = rplparallel.RPLParallel()
# print(ag.get_filename())
# ag.load(fname="rplparallel_b6ee.hkl")
# print(ag.get_filename())
# print(ag.timeStamps)
# ppg = PanGUI.create_window(pg, indexer="trial")
