from pyedfread import edfread
import numpy as np
import numpy.matlib
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import PanGUI
import DataProcessingTools as DPT

class eyelink(DPT.DPObject):
    '''
    eyelink(redoLevel=0, saveLevel=0)
    '''
    filename = 'eyelink.hkl'
    argsList = [('ObjectLevel', 'Session'), ('FileName', '.edf'), ('CalibFileNameChar', 'P'), 
    ('NavDirName', 'session'), ('ScreenX', 1920), ('ScreenY', 1080),
    ('NumTrialMessages', 3), ('TriggerMessage', 'Trigger Version 84')]

    def __init__(self, *args, **kwargs):
        # initialize fields in parent
        DPT.DPObject.__init__(self, normpath=False, *args, **kwargs)

    def create(self, *args, **kwargs):
        # initialize fields in eyelink object
        self.calib_trial_timestamps = pd.DataFrame()
        self.calib_indices = pd.DataFrame()
        self.calib_eye_pos = pd.DataFrame()
        self.calib_numSets = 0

        self.trial_timestamps = pd.DataFrame()
        self.eye_pos = pd.DataFrame()
        self.numSets = 0
        self.expTime = 0
        self.timestamps = pd.DataFrame()
        self.timeouts = pd.DataFrame()
        self.noOfTrials = 0
        self.fix_event = pd.DataFrame()
        self.fix_times = pd.DataFrame()
        self.sacc_event = pd.DataFrame()
        self.trial_codes = pd.DataFrame()
        self.session_start = 0
        self.session_start_index = 0
        
        # search current directory for .edf files
        files = os.listdir() 
        # for root, dirs, files in os.walk("."): # checks all subfolders as well

        for file in files:
            if file.endswith(self.args['FileName']):
                if file.startswith(self.args['CalibFileNameChar']):
                    # calibration file w/ format: 'Pm_d.edf'
                    print('Reading P edf file.\n')
                    # create_calib_obj(self, file)

                    samples, events, messages = pread(
                    file, trial_marker=b'1  0  0  0  0  0  0  0')

                    '''
                    Used to filter out unneeded columns of dataframes returned by pread when viewing them in the terminal.

                    samp_cols2 = ['px_left', 'px_right', 'py_left', 'py_right',
                                'hx_left', 'hx_right', 'hy_left', 'hy_right', 'pa_left',
                                'pa_right', 'gx_right', 'gy_right',
                                'rx', 'ry', 'gxvel_left', 'gxvel_right', 'gyvel_left',
                                'gyvel_right', 'hxvel_left', 'hxvel_right', 'hyvel_left',
                                'hyvel_right', 'rxvel_left', 'rxvel_right', 'ryvel_left',
                                'ryvel_right', 'fgxvel', 'fgyvel', 'fhxyvel', 'fhyvel',
                                'frxyvel', 'fryvel', 'buttons', 'flags', 'input',
                                'errors']

                    msg_cols2 = ['DISPLAY_COORDS', 'DISPLAY_COORDS_time',
                                '!CAL', '!CAL_time', 'VALIDATE', 'VALIDATE_time',
                                'RECCFG', 'RECCFG_time', 'ELCLCFG', 'ELCLCFG_time',
                                'GAZE_COORDS', 'GAZE_COORDS_time', 'THRESHOLDS',  'THRESHOLDS_time',
                                'ELCL_PROC', 'ELCL_PROC_time', 'ELCL_PCR_PARAM']

                    samples2 = samples2.drop(samp_cols2, 1)
                    messages2 = messages2.drop(msg_cols2, 1)
                    '''

                    # trial_timestamps
                    time_split = (messages['0_time']).apply(pd.Series)
                    time_split = time_split.rename(columns=lambda x: 'time_' + str(x))
                    removed = time_split['time_0'].iloc[-1]
                    # remove end value from middle column
                    time_split['time_0'] = time_split['time_0'][:-1]
                    # append removed value to last column
                    time_split['time_1'].iloc[-1] = removed
                    trial_timestamps = pd.concat(
                        [messages['trialid_time'], time_split['time_0'], time_split['time_1']], axis=1, sort=False)
                    trial_timestamps = trial_timestamps.iloc[1:]
                    # print(trial_timestamps)

                    # indices
                    index_1 = (messages['trialid_time'] - samples['time'].iloc[0]).iloc[1:]
                    index_2 = (time_split['time_0'] - samples['time'].iloc[0]).iloc[1:]
                    index_3 = (time_split['time_1'] - samples['time'].iloc[0]).iloc[1:]
                    indices = pd.concat([index_1, index_2, index_3], axis=1, sort=False)
                    # print(indices)

                    # eye_positions
                    eye_pos = samples[['gx_left', 'gy_left']].copy()
                    eye_pos['gx_left'][eye_pos['gx_left'] < 0] = np.nan
                    eye_pos['gx_left'][eye_pos['gx_left'] > self.args['ScreenX']] = np.nan
                    eye_pos['gy_left'][eye_pos['gy_left'] < 0] = np.nan
                    eye_pos['gy_left'][eye_pos['gy_left'] > self.args['ScreenY']] = np.nan
                    # print(eye_pos[1:10])
                    
                    # numSets
                    numSets = 1

                    self.calib_eye_pos = eye_pos
                    self.calib_indices = indices
                    self.calib_trial_timestamps = trial_timestamps
                    self.calib_numSets = numSets

                else:
                    # navigation file w/ format: 'yymmdd.edf'
                    print('Reading day edf file.\n')
                    #create_nav_obj(self, file)
                    samples, events, messages = pread(
                            file, trial_marker=b'Start Trial')
                    
                    # Used to filter out unneeded columns of dataframes returned by pread.
                    samp_cols = ['px_left', 'px_right', 'py_left', 'py_right',
                                'hx_left', 'hx_right', 'hy_left', 'hy_right', 'pa_left',
                                'pa_right', 'gx_right', 'gy_right',
                                'rx', 'ry', 'gxvel_left', 'gxvel_right', 'gyvel_left',
                                'gyvel_right', 'hxvel_left', 'hxvel_right', 'hyvel_left',
                                'hyvel_right', 'rxvel_left', 'rxvel_right', 'ryvel_left',
                                'ryvel_right', 'fgxvel', 'fgyvel', 'fhxyvel', 'fhyvel',
                                'frxyvel', 'fryvel', 'buttons', 'htype',
                                'errors']

                    event_cols = ['hstx', 'hsty', 'gstx', 'supd_x',
                                'gsty', 'sta', 'henx', 'heny',
                                'genx', 'geny', 'ena', 'havx',
                                'havy', 'gavx', 'gavy', 'ava',
                                'avel', 'pvel', 'svel', 'evel',
                                'eupd_x', 'eye', 'buttons', 'trial', 'blink']

                    msg_cols = ['RECCFG', 'RECCFG_time', 'ELCLCFG', 'ELCLCFG_time',
                                'GAZE_COORDS', 'GAZE_COORDS_time', 'THRESHOLDS',
                                'THRESHOLDS_time', 'ELCL_PROC', 'ELCL_PROC_time',
                                'ELCL_PCR_PARAM', 'ELCL_PCR_PARAM_time', '!MODE',
                                '!MODE_time']

                    samples = samples.drop(samp_cols, 1)
                    events = events.drop(event_cols, 1)
                    messages = messages.drop(msg_cols, 1)
                    
                    #print(samples)
                    #print(events)
                    #print(messages)

                    # expTime
                    expTime = samples['time'].iloc[0] - 1

                    # timestamps
                    timestamps = samples['time'] - 2198659

                    # eye_positions
                    eye_pos = samples[['gx_left', 'gy_left']].copy()
                    eye_pos['gx_left'][(eye_pos['gx_left'] < 0) | (eye_pos['gx_left'] > self.args['ScreenX'])] = np.nan
                    eye_pos['gy_left'][(eye_pos['gy_left'] < 0) | (eye_pos['gy_left'] > self.args['ScreenY'])] = np.nan

                    # timeout
                    timeouts = messages['Timeout_time'].dropna()

                    # noOfTrials
                    noOfTrials = len(messages) - 1

                    # fix_event
                    duration = events['end'] - events['start']
                    fix_event = duration  # difference is duration
                    fix_event = fix_event.loc[events['type'] == 'fixation']  # get fixations only
                    fix_event = fix_event.iloc[3:]  # 3 might not hold true for all files

                    # fix_times
                    fix_times = pd.concat([events['start'], events['end'],
                                        duration], axis=1, sort=False)
                    fix_times = fix_times.loc[events['type'] == 'fixation'] # get fixations only
                    fix_times['start'] = fix_times['start'] - 2198659
                    fix_times['end'] = fix_times['end'] - 2198659
                    fix_times = fix_times.iloc[3:]
                    # print(fix_times)

                    # sacc_event
                    sacc_event = events['end'] - events['start']  # difference is duration
                    sacc_event = sacc_event.loc[events['type'] == 'saccade']  # get fixations only
                    sacc_event = sacc_event.iloc[3:]

                    # numSets
                    numSets = 1

                    # trial_timestamps
                    timestamps_1 = messages['trialid_time'] - expTime
                    timestamps_2 = messages['Cue_time'] - expTime
                    timestamps_3 = messages['End_time'] - expTime
                    trial_timestamps = pd.concat(
                        [timestamps_1, timestamps_2, timestamps_3], axis=1, sort=False)
                    trial_timestamps = trial_timestamps.iloc[1:]
                    # print(trial_timestamps)

                    # trial_codes
                    trial_id = messages['trialid '].str.replace(
                        r'\D', '')  # remove 'Start Trial ' from string
                    cue_split = messages['Cue'].apply(pd.Series)
                    cue_split = cue_split.rename(columns=lambda x: 'cue_' + str(x))
                    end_split = messages['End'].apply(pd.Series)
                    end_split = end_split.rename(columns=lambda x: 'end_' + str(x))
                    trial_codes = pd.concat(
                        [trial_id, cue_split['cue_1'], end_split['end_1']], axis=1, sort=False)
                    trial_codes = trial_codes.iloc[1:]
                    trial_codes = trial_codes.astype(np.float64)  # convert all columns into float dt

                    # session_start
                    samples2, events2, messages2 = pread(
                        file, trial_marker=b'Trigger Version 84')
                    session_start = messages2['trialid_time'].iloc[1]
                    
                    messages2 = messages2.drop(msg_cols, 1)
                    # print(messages2)
                    # session_start_index
                    session_start_index = session_start - expTime
                    # print(session_start_index)

                    self.expTime = expTime
                    self.timestamps = timestamps
                    self.eye_pos = eye_pos
                    self.timeouts = timeouts
                    self.noOfTrials = noOfTrials
                    self.fix_event = fix_event
                    self.fix_times = fix_times
                    self.sacc_event = sacc_event
                    self.numSets = numSets
                    self.trial_timestamps = trial_timestamps
                    self.trial_codes = trial_codes
                    self.session_start = session_start
                    self.session_start_index = session_start_index

        # check if we need to save the object
        if kwargs.get("saveLevel", 0) > 0:
            self.save()

    def append(self, df):
        # update fields in parent
        DPT.DPObject.append(self, df)

        # update fields in child
        self.calib_trial_timestamps += df.calib_trial_timestamps
        self.calib_indices += df.calib_indices
        self.calib_eye_pos += df.calib_eye_pos
        self.calib_numSets += df.calib_numSets

        self.trial_timestamps += df.trial_timestamps
        self.eye_pos += df.eye_pos
        self.numSets += df.numSets
        self.expTime += df.expTime
        self.timestamps += df.timestamps
        self.timeouts += df.timeouts
        self.noOfTrials += df.noOfTrials
        self.fix_event += df.fix_event
        self.fix_times += df.fix_times
        self.sacc_event += df.sacc_event
        self.trial_codes += df.trial_codes
        self.session_start += df.session_start
        self.session_start_index += df.session_start_index

    def plot(self, i=None, ax=None, overlay=False):
        self.current_idx = i
        if ax is None:
            ax = plt.gca()
        if not overlay:
            ax.clear()
        # print(i)

        # set plot options
        self.plotopts = {"Type": DPT.objects.ExclusiveOptions(['Trial', 'XY', 'Calibration', 'CalTrial', 'SaccFix'],0)}
        plot_type = self.plotopts['Type'].selected()

        # Trial - Plot x vs t and y vs t positions per trial
        if (plot_type == 'Trial'):
            x = self.trial_timestamps.to_numpy()
            obj_timestamps = self.timestamps.to_numpy()

            trial_start_time = obj_timestamps[x[i][0].astype(int)]
            trial_cue_time = obj_timestamps[x[i][1].astype(int)] - trial_start_time - 1
            trial_end_time = obj_timestamps[x[i][2].astype(int)] - 1

            # timestamps is the x axis to be plotted
            timestamps = obj_timestamps[x[i][0].astype(int) - 501 : x[i][2].astype(int)]
            timestamps = timestamps - trial_start_time
            obj_eye_pos = self.eye_pos.to_numpy()
            y = obj_eye_pos[x[i][0].astype(int) - 501 : x[i][2].astype(int)].transpose()
            
            # plot x axis data
            ax.plot(timestamps, y[:][0], 'b-', LineWidth=0.5, Label='X position')
            # label axis
            ax.set_title('Eye Movements versus Time for Trial ' + str(i))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Position (screen pixels)')
            # plot y axis
            ax.plot(timestamps, y[:][1], 'g-', LineWidth=0.5, Label='Y position')
            
            # Plotting lines to mark the start, cue offset, and end/timeout for the trial
            ax.plot([0, 0], ax.set_ylim(), 'g', LineWidth=0.5)
            ax.plot([trial_cue_time, trial_cue_time], ax.set_ylim(), 'm', LineWidth=0.5)
            timedOut = self.timeouts == trial_end_time
            trial_end_time = trial_end_time - trial_start_time
            timedOut = np.nonzero(timedOut.to_numpy)

            if not timedOut: # trial did not timeout
                ax.plot([trial_end_time, trial_end_time], ax.set_ylim(), 'b', LineWidth=0.5)
            else: # trial did timeout
                ax.plot([trial_end_time, trial_end_time], ax.set_ylim(), 'r', LineWidth=0.5)

            ax.set_xlim([-100, trial_end_time + 100]) # set axis boundaries
            ax.set_ylim([0, 1800])
            ax.legend(loc='best')

        elif (plot_type == 'XY'):
            # XY - Plots the x and y movement of the eye per trial extract all the trials from one session 
            n = i[0]
            x = self.data.trial_timestamps.to_numpy()
            obj_eye_pos = self.data.eye_pos.to_numpy()
            y = obj_eye_pos[x[n][0].astype(int) : x[n][2].astype(int), :].transpose()
            ax = plotGazeXY(ax, y[0], y[1], 'b') # plot blue circles

        elif (plot_type == 'Calibration'):
            # Calibration - Plot of calibration eye movements
            n = i[0]
            obj_eye_pos = self.data.calib_eye_pos.to_numpy()
            indices = self.data.calib_indices.to_numpy()
            y = obj_eye_pos[indices[n][0].astype(int) : indices[n][2].astype(int), :]
            y = y.transpose()
            ax = plotGazeXY(y[0], y[1], 'b')

        elif (plot_type == 'CalTrial'):
            # CalTrial
            n = i[0]
            obj_eye_pos = self.data.calib_eye_pos.to_numpy()
            indices = self.data.calib_indices.to_numpy()
            y = obj_eye_pos[indices[n][0].astype(int) : indices[n][2].astype(int), :]
            ax.plot(y, 'o-', fillstyle='none')
            # lines.Line2D(np.matlib.repmat(obj_eye_pos[indices[n][1].astype(int)], 1, 2), ax.set_ylim())
            ax.set_ylim([0, 1200])
            ax.set_xlim([0, 3500])

        elif (plot_type == 'SaccFix'):
            # SaccFix - Histogram of fixations and saccades per session
            n = i[0]
            sacc_durations = self.data.sacc_event[:][n].to_numpy()
            fix_durations = self.data.fix_event[:][n].to_numpy()
            
            sacc_durations = sacc_durations[sacc_durations != 0]
            fix_durations = fix_durations[fix_durations != 0]

            lower = np.amin(fix_durations)
            upper = np.amax(fix_durations)
            
            edges = np.arange(lower, upper, 25).tolist()
            edges = [x for x in edges if x <= 1000]

            ax.hist(sacc_durations, density=False, alpha=0.5, color='#31b4e8', bins=edges, label='Saccades', edgecolor='black', linewidth=0.3)
            ax.hist(fix_durations, density=False, alpha=0.5, color='#ed7f18', bins=edges, label='Fixations', edgecolor='black', linewidth=0.3)
            ax.set_title ('Distribution of Saccades and Fixations for the Session')
            ax.set_xlabel ('Duration (ms)')
            ax.set_ylabel ('# of events')
            ax.legend(loc='best')
        
        return ax
        
# plotGazeXY helper method to plot gaze position. Uses matlab's plot function
def plotGazeXY(ax, gx, gy, lineType):
    ax.scatter(gx, gy, color='none', edgecolor=lineType)
    ax.invert_yaxis() # reverse current y axis

    #currentAxis = ax.gca() # draw rect to represent the screen
    ax.add_patch(patches.Rectangle((0, 0), 1920, 1080, fill=None, alpha=1, lw=0.5))

    ax.set_title('Calibration Eye movements from session')
    ax.set_xlabel('Gaze Position X (screen pixels)')
    ax.set_ylabel(('Gaze Position Y (screen pixels)'))
    ax.set_xlim([0, 2000]) # set axis boundaries, use max
    ax.set_ylim([1300, 0])
    ax.legend(loc='best')

    return ax

######### Calibration file #########
def create_calib_obj(el, fileName):

    samples, events, messages = pread(
        fileName, trial_marker=b'1  0  0  0  0  0  0  0')

    '''
    Used to filter out unneeded columns of dataframes returned by pread when viewing them in the terminal.

    samp_cols2 = ['px_left', 'px_right', 'py_left', 'py_right',
                'hx_left', 'hx_right', 'hy_left', 'hy_right', 'pa_left',
                'pa_right', 'gx_right', 'gy_right',
                'rx', 'ry', 'gxvel_left', 'gxvel_right', 'gyvel_left',
                'gyvel_right', 'hxvel_left', 'hxvel_right', 'hyvel_left',
                'hyvel_right', 'rxvel_left', 'rxvel_right', 'ryvel_left',
                'ryvel_right', 'fgxvel', 'fgyvel', 'fhxyvel', 'fhyvel',
                'frxyvel', 'fryvel', 'buttons', 'flags', 'input',
                'errors']

    msg_cols2 = ['DISPLAY_COORDS', 'DISPLAY_COORDS_time',
                '!CAL', '!CAL_time', 'VALIDATE', 'VALIDATE_time',
                'RECCFG', 'RECCFG_time', 'ELCLCFG', 'ELCLCFG_time',
                'GAZE_COORDS', 'GAZE_COORDS_time', 'THRESHOLDS',  'THRESHOLDS_time',
                'ELCL_PROC', 'ELCL_PROC_time', 'ELCL_PCR_PARAM']

    samples2 = samples2.drop(samp_cols2, 1)
    messages2 = messages2.drop(msg_cols2, 1)
    '''

    # trial_timestamps
    time_split = (messages['0_time']).apply(pd.Series)
    time_split = time_split.rename(columns=lambda x: 'time_' + str(x))
    removed = time_split['time_0'].iloc[-1]
    # remove end value from middle column
    time_split['time_0'] = time_split['time_0'][:-1]
    # append removed value to last column
    time_split['time_1'].iloc[-1] = removed
    trial_timestamps = pd.concat(
        [messages['trialid_time'], time_split['time_0'], time_split['time_1']], axis=1, sort=False)
    trial_timestamps = trial_timestamps.iloc[1:]
    # print(trial_timestamps)

    # indices
    index_1 = (messages['trialid_time'] - samples['time'].iloc[0]).iloc[1:]
    index_2 = (time_split['time_0'] - samples['time'].iloc[0]).iloc[1:]
    index_3 = (time_split['time_1'] - samples['time'].iloc[0]).iloc[1:]
    indices = pd.concat([index_1, index_2, index_3], axis=1, sort=False)
    # print(indices)

    # eye_positions
    eye_pos = samples[['gx_left', 'gy_left']].copy()
    eye_pos['gx_left'][eye_pos['gx_left'] < 0] = np.nan
    eye_pos['gx_left'][eye_pos['gx_left'] > el.args['ScreenX']] = np.nan
    eye_pos['gy_left'][eye_pos['gy_left'] < 0] = np.nan
    eye_pos['gy_left'][eye_pos['gy_left'] > el.args['ScreenY']] = np.nan
    # print(eye_pos[1:10])
    
    # numSets
    numSets = 1

    el.calib_eye_pos = eye_pos
    el.calib_indices = indices
    el.calib_trial_timestamps = trial_timestamps
    el.calib_numSets = numSets

    return el

######### Navigation file #########
def create_nav_obj(el, fileName):

    samples, events, messages = pread(
        fileName, trial_marker=b'Start Trial')
 
    # Used to filter out unneeded columns of dataframes returned by pread.
    samp_cols = ['px_left', 'px_right', 'py_left', 'py_right',
                 'hx_left', 'hx_right', 'hy_left', 'hy_right', 'pa_left',
                 'pa_right', 'gx_right', 'gy_right',
                 'rx', 'ry', 'gxvel_left', 'gxvel_right', 'gyvel_left',
                 'gyvel_right', 'hxvel_left', 'hxvel_right', 'hyvel_left',
                 'hyvel_right', 'rxvel_left', 'rxvel_right', 'ryvel_left',
                 'ryvel_right', 'fgxvel', 'fgyvel', 'fhxyvel', 'fhyvel',
                 'frxyvel', 'fryvel', 'buttons', 'htype',
                 'errors']

    event_cols = ['hstx', 'hsty', 'gstx', 'supd_x',
                  'gsty', 'sta', 'henx', 'heny',
                  'genx', 'geny', 'ena', 'havx',
                  'havy', 'gavx', 'gavy', 'ava',
                  'avel', 'pvel', 'svel', 'evel',
                  'eupd_x', 'eye', 'buttons', 'trial', 'blink']

    msg_cols = ['RECCFG', 'RECCFG_time', 'ELCLCFG', 'ELCLCFG_time',
                'GAZE_COORDS', 'GAZE_COORDS_time', 'THRESHOLDS',
                'THRESHOLDS_time', 'ELCL_PROC', 'ELCL_PROC_time',
                'ELCL_PCR_PARAM', 'ELCL_PCR_PARAM_time', '!MODE',
                '!MODE_time']

    samples = samples.drop(samp_cols, 1)
    events = events.drop(event_cols, 1)
    messages = messages.drop(msg_cols, 1)
    
    #print(samples)
    #print(events)
    #print(messages)

    # expTime
    expTime = samples['time'].iloc[0] - 1

    # timestamps
    timestamps = samples['time'] - 2198659

    # eye_positions
    eye_pos = samples[['gx_left', 'gy_left']].copy()
    eye_pos['gx_left'][(eye_pos['gx_left'] < 0) | (eye_pos['gx_left'] > el.args['ScreenX'])] = np.nan
    eye_pos['gy_left'][(eye_pos['gy_left'] < 0) | (eye_pos['gy_left'] > el.args['ScreenY'])] = np.nan

    # timeout
    timeouts = messages['Timeout_time'].dropna()

    # noOfTrials
    noOfTrials = len(messages) - 1

    # fix_event
    duration = events['end'] - events['start']
    fix_event = duration  # difference is duration
    fix_event = fix_event.loc[events['type'] == 'fixation']  # get fixations only
    fix_event = fix_event.iloc[3:]  # 3 might not hold true for all files

    # fix_times
    fix_times = pd.concat([events['start'], events['end'],
                           duration], axis=1, sort=False)
    fix_times = fix_times.loc[events['type'] == 'fixation'] # get fixations only
    fix_times['start'] = fix_times['start'] - 2198659
    fix_times['end'] = fix_times['end'] - 2198659
    fix_times = fix_times.iloc[3:]
    # print(fix_times)

    # sacc_event
    sacc_event = events['end'] - events['start']  # difference is duration
    sacc_event = sacc_event.loc[events['type'] == 'saccade']  # get fixations only
    sacc_event = sacc_event.iloc[3:]

    # numSets
    numSets = 1

    # trial_timestamps
    timestamps_1 = messages['trialid_time'] - expTime
    timestamps_2 = messages['Cue_time'] - expTime
    timestamps_3 = messages['End_time'] - expTime
    trial_timestamps = pd.concat(
        [timestamps_1, timestamps_2, timestamps_3], axis=1, sort=False)
    trial_timestamps = trial_timestamps.iloc[1:]
    # print(trial_timestamps)

    # trial_codes
    trial_id = messages['trialid '].str.replace(
        r'\D', '')  # remove 'Start Trial ' from string
    cue_split = messages['Cue'].apply(pd.Series)
    cue_split = cue_split.rename(columns=lambda x: 'cue_' + str(x))
    end_split = messages['End'].apply(pd.Series)
    end_split = end_split.rename(columns=lambda x: 'end_' + str(x))
    trial_codes = pd.concat(
        [trial_id, cue_split['cue_1'], end_split['end_1']], axis=1, sort=False)
    trial_codes = trial_codes.iloc[1:]
    trial_codes = trial_codes.astype(np.float64)  # convert all columns into float dt

    # session_start
    samples2, events2, messages2 = pread(
        fileName, trial_marker=b'Trigger Version 84')
    session_start = messages2['trialid_time'].iloc[1]
    
    messages2 = messages2.drop(msg_cols, 1)
    # print(messages2)
    # session_start_index
    session_start_index = session_start - expTime
    # print(session_start_index)

    el.expTime = expTime
    el.timestamps = timestamps
    el.eye_pos = eye_pos
    el.timeouts = timeouts
    el.noOfTrials = noOfTrials
    el.fix_event = fix_event
    el.fix_times = fix_times
    el.sacc_event = sacc_event
    el.numSets = numSets
    el.trial_timestamps = trial_timestamps
    el.trial_codes = trial_codes
    el.session_start = session_start
    el.session_start_index = session_start_index

    return el

def pread(filename,
          ignore_samples=False,
          filter='all',
          split_char=' ',
          trial_marker=b'TRIALID',
          meta={}):
    '''
    Parse an EDF file into a pandas.DataFrame.
    EDF files contain three types of data: samples, events and messages.
    pread returns one pandas DataFrame for each type of information.
    '''
    if not os.path.isfile(filename):
        raise RuntimeError('File "%s" does not exist' % filename)

    if pd is None:
        raise RuntimeError('Can not import pandas.')

    samples, events, messages = edfread.fread(
        filename, ignore_samples,
        filter, split_char, trial_marker)
    events = pd.DataFrame(events)
    messages = pd.DataFrame(messages)
    samples = pd.DataFrame(np.asarray(samples), columns=edfread.sample_columns)

    for key, value in meta.items():
        events.insert(0, key, value)
        messages.insert(0, key, value)
        samples.insert(0, key, value)

    return samples, events, messages