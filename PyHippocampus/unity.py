import DataProcessingTools as DPT
from pylab import gcf, gca
import matplotlib.patches as patches
import numpy as np
from scipy.stats import iqr
import os
import glob
import networkx as nx
from scipy.spatial.distance import cdist
from .rplparallel import RPLParallel

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
    argsList = [("FileLineOffset", 15), ("DirName", "RawData*"), ("FileName", "session*"), ("TriggerVal1", 10),
                ("TriggerVal2", 20), ("TriggerVal3", 30), ("BinNumberLimit", 500)]
    level = "session"

    def __init__(self, *args, **kwargs):
        DPT.DPObject.__init__(self, *args, **kwargs)

    def create(self, *args, **kwargs):

        # look for RawData_T * folder
        if bool(glob.glob(self.args["DirName"])):
            # create object
            DPT.DPObject.create(self, *args, **kwargs)

            # set plot options
            self.indexer = self.getindex("trial")

            # initialization
            self.numSets = 0
            self.sumCost = []
            self.unityData = []
            self.unityTriggers = []
            self.unityTrialTime = []
            self.unityTime = []
            self.timePerformance = []
            self.routePerformance = []
            self.trialRouteRatio = []
            self.durationDiff = []

            # load the rplparallel object to get the Ripple timestamps
            rl = RPLParallel()
            self.timeStamps = [rl.timeStamps]

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
                num_within_time = (np.where((text_data[:, 0] > self.args["TriggerVal3"]) & (text_data[:, 0] < 40)))[0].size
                # Check if there is any incomplete trial
                utRows = [uT1[0].size, uT2[0].size, uT3[0].size]
                utMax = max(utRows)
                utMin = min(utRows)
                incomplete_trials = utMax - utMin
                if incomplete_trials != 0:
                    print("Incomplete session! Last", incomplete_trials, "trial discarded")
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
                sumCost = np.zeros((totTrials, 6))

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

                # Proportion of trial
                ratio_within_time = num_within_time / totTrials
                ratio_shortest_route = np.where(sumCost[:, 5] == 1)[0].size / totTrials
                ratio_each_trial_route = np.divide(sumCost[:, 1], sumCost[:, 0])

                # duration_diff
                start_ind = unityTriggers[:, 0] + 1
                end_ind = unityTriggers[:, 2] + 1
                start_time = unityTime[start_ind]
                end_time = unityTime[end_ind]
                trial_durations = end_time - start_time

                duration_diff = None
                try:
                    rp_trial_dur = rl.timeStamps[:, 2] - rl.timeStamps[:, 0]
                    # multiply by 1000 to convert to ms
                    duration_diff = (trial_durations - rp_trial_dur) * 1000
                except:
                    print('problem with timeStamps')

                self.durationDiff = [duration_diff]
                self.trialRouteRatio = [ratio_each_trial_route]
                self.timePerformance = [ratio_within_time]
                self.routePerformance = [ratio_shortest_route]
                self.sumCost.append(sumCost)
                self.unityData.append(unityData)
                self.unityTriggers.append(unityTriggers)
                self.unityTrialTime.append(unityTrialTime)
                self.unityTime.append(unityTime)
                self.setidx = ([0] * unityTriggers.shape[0])
        else:
            # create empty object
            DPT.DPObject.create(self, dirs=[], *args, **kwargs)

    def plot(self, i=None, getNumEvents=False, getLevels=False, getPlotOpts=False, ax=None, preOpt=None, **kwargs):
        # set plot options
        plotopts = {"PlotType": DPT.objects.ExclusiveOptions(["X-Y", "X-T", "Y-T", "Theta-T", "Frame Intervals", "Duration Diffs",
                                                                "Route Ratio", "Routes", "Proportion of trials"], 0),
                    "Frame Interval Triggers": {"from": 1.0, "to": 2.0}, "Number of bins": 0}
        if getPlotOpts:
            return plotopts

        # Extract the recognized plot options from kwargs
        for (k, v) in plotopts.items():
            plotopts[k] = kwargs.get(k, v)

        plot_type = plotopts["PlotType"].selected()

        pre = 'trial'
        if preOpt is not None:
            if preOpt == "X-Y" or preOpt == "Frame Intervals" or preOpt == "X-T" or preOpt == "Y-T" \
                    or preOpt == "Theta-T":
                pre = "trial"
            elif preOpt == "Duration Diffs" or preOpt == "Route Ratio" or preOpt == "Routes":
                pre = "session"

        if getNumEvents:
            # Return the number of events available
            # global pre
            if plot_type == "X-Y" or plot_type == "Frame Intervals" or plot_type == "X-T" or plot_type == "Y-T" \
                    or plot_type == "Theta-T":
                if i is not None:
                    if pre == "trial":
                        return len(self.setidx), i
                    else:
                        num_idx = 0
                        for x in range(0, i):
                            num_idx += self.unityTriggers[x].shape[0]
                else:
                    num_idx = 0
                return len(self.setidx), num_idx
            elif plot_type == "Duration Diffs" or plot_type == "Route Ratio" or plot_type == "Routes":
                if i is not None:
                    if pre == "session":
                        return np.max(self.setidx) + 1, i
                    else:
                        num_idx = self.setidx[i]
                else:
                    num_idx = 0
                return np.max(self.setidx) + 1, num_idx
            elif plot_type == "Proportion of trials":
                return 1, 0

        if getLevels:
            # Return the possible levels for this object
            return ["trial", "session"]

        if ax is None:
            ax = gca()

        ax.clear()
        for other_ax in ax.figure.axes:
            if other_ax is ax:
                continue
            if other_ax.bbox.bounds == ax.bbox.bounds:
                other_ax.remove()

        if plot_type == "X-Y":

            session_idx = self.setidx[i]
            if session_idx != 0:
                for x in range(0, session_idx):
                    i = i - self.unityTriggers[x].shape[0]

            ax.plot(xBound, zBound, color='k', linewidth=1.5)
            ax.plot(x1Bound, z1Bound, 'y', linewidth=1)
            ax.plot(x2Bound, z2Bound, 'r', linewidth=1)
            ax.plot(x3Bound, z3Bound, 'b', linewidth=1)
            ax.plot(x4Bound, z4Bound, 'g', linewidth=1)
            ax.text(poster_pos[0,0],poster_pos[0,1]-1,'1')
            ax.text(poster_pos[1,0]-0.5,poster_pos[1,1],'2')
            ax.text(poster_pos[2,0],poster_pos[2,1],'3')
            ax.text(poster_pos[3,0],poster_pos[3,1],'4')
            ax.text(poster_pos[4,0],poster_pos[4,1]-1,'5')
            ax.text(poster_pos[5,0],poster_pos[5,1],'6')
            x_data = self.unityData[session_idx][int(self.unityTriggers[session_idx][i, 1]):
                                                 int(self.unityTriggers[session_idx][i, 2]), 2]
            y_data = self.unityData[session_idx][int(self.unityTriggers[session_idx][i, 1]):
                                                 int(self.unityTriggers[session_idx][i, 2]), 3]
            ax.plot(x_data, y_data, 'b+', linewidth=1)

            # plot end point identifier
            ax.plot(self.unityData[session_idx][self.unityTriggers[session_idx][i, 2], 2],
                    self.unityData[session_idx][self.unityTriggers[session_idx][i, 2], 3], 'k.', markersize=10)
            route_str = str(self.sumCost[session_idx][i, 1])
            short_str = str(self.sumCost[session_idx][i, 0])
            ratio_str = str(self.sumCost[session_idx][i, 1] / self.sumCost[session_idx][i, 0])
            title = ' T: ' + str(i) + ' Route: ' + route_str + ' Shortest: ' + short_str + ' Ratio: ' + ratio_str

            dir_name = self.dirs[session_idx]
            subject = DPT.levels.get_shortname("subject", dir_name)
            date = DPT.levels.get_shortname("day", dir_name)
            session = DPT.levels.get_shortname("session", dir_name)
            title = subject + date + session + title
            ax.set_title(title)

        elif plot_type == "Frame Intervals":

            session_idx = self.setidx[i]
            if session_idx != 0:
                for x in range(0, session_idx):
                    i = i - self.unityTriggers[x].shape[0]

            time_stamps = self.timeStamps[session_idx]
            frame_interval_triggers = np.array([plotopts["Frame Interval Triggers"]["from"],
                                                plotopts["Frame Interval Triggers"]["to"]], dtype=np.int)
            indices = self.unityTriggers[session_idx][i, frame_interval_triggers]
            u_data = self.unityData[session_idx][(indices[0] + 1):(indices[1] + 1), 1]
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

            dir_name = self.dirs[session_idx]
            subject = DPT.levels.get_shortname("subject", dir_name)
            date = DPT.levels.get_shortname("day", dir_name)
            session = DPT.levels.get_shortname("session", dir_name)
            title = subject + date + session + title
            ax.set_title(title)

        elif plot_type == "Duration Diffs":

            if plotopts["Number of bins"] == 0:
                # use The Freedman-Diaconis rule to get optimal bin-width
                tot_num = self.durationDiff[i].shape[0]
                num_range = np.amax(self.durationDiff[i]) - np.amin(self.durationDiff[i])
                IQR = iqr(self.durationDiff[i])
                bin_width = 2 * IQR / pow(tot_num, 1/3)
                num_bin = num_range / bin_width
                if num_bin > self.args["BinNumberLimit"]:
                    num_bin = self.args["BinNumberLimit"]
            else:
                num_bin = plotopts["Number of bins"]

            ax.hist(x=self.durationDiff[i], bins=int(num_bin))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Frequency')
            ax.set_yscale("log")
            ax.grid(axis="y")

            dir_name = self.dirs[i]
            subject = DPT.levels.get_shortname("subject", dir_name)
            date = DPT.levels.get_shortname("day", dir_name)
            session = DPT.levels.get_shortname("session", dir_name)
            ax.set_title('Unity trial duration - Ripple trial duration ' + subject + date + session)

        elif plot_type == "Route Ratio":
            tot_trials = self.unityTriggers[i].shape[0]
            xind = np.arange(0, tot_trials)
            # Calculate optimal width
            width = np.min(np.diff(xind)) / 3
            ax.bar(xind - width / 2, self.sumCost[i][:, 0], width, color='yellow', label="Shortest")
            ax.bar(xind + width / 2, self.sumCost[i][:, 1], width, color='cyan', label="Route")
            ax1 = ax.twinx()
            # ratio = np.divide(self.sumCost[session_idx][:, 1], self.sumCost[session_idx][:, 0])
            # markerline, stemlines, baseline = ax1.stem(xind, ratio, 'magenta', markerfmt='mo', basefmt=" ",
            #                                            use_line_collection=True, label='Ratio')
            markerline, stemlines, baseline = ax1.stem(xind, self.trialRouteRatio[i], 'magenta',
                                                       markerfmt='mo', basefmt=" ", use_line_collection=True,
                                                       label='Ratio')
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

            dir_name = self.dirs[i]
            subject = DPT.levels.get_shortname("subject", dir_name)
            date = DPT.levels.get_shortname("day", dir_name)
            session = DPT.levels.get_shortname("session", dir_name)
            ax.set_title(subject + date + session)

        elif plot_type == "Proportion of trials":

            session_num = np.arange(0, len(self.unityData))
            ax.plot(session_num, self.timePerformance, label='Completed within time limit',
                    marker='o', fillstyle='none')
            ax.plot(session_num, self.routePerformance, label='Completed within time limit and via the shortest route',
                    marker='o', fillstyle='none')
            ax.set_xlabel('Session')
            ax.set_ylabel('Proportion of trials')
            ax.legend(loc="lower center")
            ax.set_ylim(0, 1.2)

        elif plot_type == "Routes":

            # add grid
            for a in range(0, 2):
                ax.plot([x1Bound[a], x1Bound[a]], xBound[0:2], color='gray', linewidth=0.5)
                ax.plot([x2Bound[a], x2Bound[a]], xBound[0:2], color='gray', linewidth=0.5)
                ax.plot(xBound[0:2], [x1Bound[a], x1Bound[a]], color='gray', linewidth=0.5)
                ax.plot(xBound[0:2], [x2Bound[a], x2Bound[a]], color='gray', linewidth=0.5)
            # bound
            ax.plot(xBound, zBound, color='k', linewidth=1.5)
            rect1 = patches.Rectangle((x1Bound[0], z1Bound[0]), 5, -5, linewidth=1, edgecolor='k', facecolor='k')
            rect2 = patches.Rectangle((x2Bound[0], z2Bound[0]), 5, -5, linewidth=1, edgecolor='k', facecolor='k')
            rect3 = patches.Rectangle((x3Bound[0], z3Bound[0]), 5, -5, linewidth=1, edgecolor='k', facecolor='k')
            rect4 = patches.Rectangle((x4Bound[0], z4Bound[0]), 5, -5, linewidth=1, edgecolor='k', facecolor='k')
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            ax.add_patch(rect3)
            ax.add_patch(rect4)
            # add poster
            # ax.plot(poster_pos[:, 0], poster_pos[:, 1], 'o', color='r')
            ax.text(poster_pos[0,0],poster_pos[0,1]-1,'1')
            ax.text(poster_pos[1,0]-0.5,poster_pos[1,1],'2')
            ax.text(poster_pos[2,0],poster_pos[2,1],'3')
            ax.text(poster_pos[3,0],poster_pos[3,1],'4')
            ax.text(poster_pos[4,0],poster_pos[4,1]-1,'5')
            ax.text(poster_pos[5,0],poster_pos[5,1],'6')

            x_data = self.unityData[i][:, 2]
            y_data = self.unityData[i][:, 3]
            ax.plot(x_data, y_data, linewidth=1)

            dir_name = self.dirs[i]
            subject = DPT.levels.get_shortname("subject", dir_name)
            date = DPT.levels.get_shortname("day", dir_name)
            session = DPT.levels.get_shortname("session", dir_name)
            title = "Routes: " + subject + date + session
            ax.set_title(title)

        elif plot_type == "X-T":
            i, session_idx, trial_trigger = self.get_trial_trigger(i)
            x_position = self.unityData[session_idx][trial_trigger[0]:trial_trigger[1]+1, 2]   
            self.get_timestamps(i, session_idx, trial_trigger, ax, x_position)
            
            ax.set_ylabel('X-Pos')
            ax.set_xlabel('Time (s)')

            self.set_T_title(i, session_idx, ax)

        elif plot_type == "Y-T":
            i, session_idx, trial_trigger = self.get_trial_trigger(i)
            y_position = self.unityData[session_idx][trial_trigger[0]:trial_trigger[1]+1, 3]
            self.get_timestamps(i, session_idx, trial_trigger, ax, y_position) 

            ax.set_ylabel('Y-Pos')
            ax.set_xlabel('Time (s)')

            self.set_T_title(i, session_idx, ax)


        elif plot_type == "Theta-T":
            i, session_idx, trial_trigger = self.get_trial_trigger(i)
            orientation = self.unityData[session_idx][trial_trigger[0]:trial_trigger[1]+1, 4]
            self.get_timestamps(i, session_idx, trial_trigger, ax, orientation)

            ax.set_ylabel('Orientation')
            ax.set_xlabel('Time (s)')

            self.set_T_title(i, session_idx, ax)


        return ax

    def append(self, uf):
        # update fields in parent
        DPT.DPObject.append(self, uf)
        # update fields in child
        self.numSets += uf.numSets
        self.sumCost += uf.sumCost
        self.unityData += uf.unityData
        self.unityTriggers += uf.unityTriggers
        self.unityTrialTime += uf.unityTrialTime
        self.unityTime += uf.unityTime
        self.timeStamps += uf.timeStamps
        self.timePerformance += uf.timePerformance
        self.routePerformance += uf.routePerformance
        self.trialRouteRatio += uf.trialRouteRatio
        self.durationDiff += uf.durationDiff
        
    def get_trial_trigger(self, i):
        session_idx = self.setidx[i]        
        if session_idx != 0:
            for x in range(0, session_idx):
                i = i - self.unityTriggers[x].shape[0]
        if i == 0:
            trial_trigger = [0, self.unityTriggers[session_idx][i+1, 0]-1]
        elif i == self.unityTriggers[session_idx].shape[0]-1:
            trial_trigger = [self.unityTriggers[session_idx][i-1, 2]+1, self.unityData[session_idx].shape[0]-1]
        else:
            trial_trigger = [self.unityTriggers[session_idx][i-1, 2]+1, self.unityTriggers[session_idx][i+1, 0]-1]
        return i, session_idx, trial_trigger
        
    def get_timestamps(self, i, session_idx, trial_trigger, ax, data):
        time_start = self.unityTime[session_idx][self.unityTriggers[session_idx][i, 0]+1]  # original start time
        time_cue = self.unityTime[session_idx][self.unityTriggers[session_idx][i, 1]+1]  # original cue time
        time_end = self.unityTime[session_idx][self.unityTriggers[session_idx][i, 2]+1]  # original end time
        
        time = self.unityTime[session_idx][trial_trigger[0]+1:trial_trigger[1]+2]  # original data time
        time_shift = time[0]  # for shifting markers to 0
        
        t_1 = time_start - time_shift  # start marker shifted accordingly to the first data timestamps
        t_2 = time_cue - time_shift  # cue marker shifted accordingly to the first data timestamps 
        t_3 = time_end - time_shift  # end marker shifted accordingly to the first data timestamps

        time = time - time_shift - t_1  # shift data time to 0 first, then shift it again according to start marker, because this data starts at the end marker of previous trial, i.e. it will start wiht negative value
        
        time_shift = t_1  # for shifting start marker to 0
        t_1 -= time_shift
        t_2 -= time_shift
        t_3 -= time_shift  
        
        result_t3 = self.unityData[session_idx][self.unityTriggers[session_idx][i, 2], 0]
        
        ax.plot(time, data, linewidth=1)

        if result_t3 > 40:
            ax.axvline(t_3, color='r')
        else:
            ax.axvline(t_3, color='b')
        ax.axvline(t_1, color='g')
        ax.axvline(t_2, color='m')        
    
    def set_T_title(self, i, session_idx, ax):
        dir_name = self.dirs[session_idx]
        subject = DPT.levels.get_shortname("subject", dir_name)
        date = DPT.levels.get_shortname("day", dir_name)
        session = DPT.levels.get_shortname("session", dir_name)
        title = subject + date + session + " Trial " + str(i)
        ax.set_title(title)