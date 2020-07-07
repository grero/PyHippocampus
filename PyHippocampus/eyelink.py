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

Args = {'RedoLevels': 0, 'SaveLevels': 0, 'Auto': 0, 'ArgsOnly': 0, 'ObjectLevel': 'Session',
        'FileName': '.edf', 'CalibFileNameChar': 'P', 'EventTypeNum': 24,
        'NavDirName': 'session0*', 'SessionEyeName': 'sessioneye',
        'ScreenX': 1920, 'ScreenY': 1080, 'NumMessagesToClear': 7,
        'NumTrialMessages': 3, 'TriggerMessage': 'Trigger Version 84'}

class eyelink():
    def __init__(self, redoLevels=0, saveLevels=0, objectLevel='Session'):
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

    def save(self):
        hf = h5py.File('eyelink.hdf5', mode='w')

        hf.create_dataset('calib_numSets', data=el.calib_numSets)
        hf.create_dataset('calib_trial_timestamps', data=el.calib_trial_timestamps)
        hf.create_dataset('calib_indices', data=el.calib_indices)
        hf.create_dataset('calib_eye_pos', data=el.calib_eye_pos)

        hf.create_dataset('numSets', data=el.numSets)
        hf.create_dataset('trial_timestamps', data=el.trial_timestamps)
        hf.create_dataset('eye_pos', data=el.eye_pos)
        hf.create_dataset('expTime', data=el.expTime)
        hf.create_dataset('timestamps', data=el.timestamps)
        hf.create_dataset('timeouts', data=el.timeouts)
        hf.create_dataset('noOfTrials', data=el.noOfTrials)
        hf.create_dataset('fix_event', data=el.fix_event)
        hf.create_dataset('fix_times', data=el.fix_times)
        hf.create_dataset('sacc_event', data=el.sacc_event)
        hf.create_dataset('trial_codes', data=el.trial_codes)
        hf.create_dataset('session_start', data=el.session_start)
        hf.create_dataset('session_start_index', data=el.session_start_index)

        hf.close()

    def plot(self):
        obj = plot(self)

class plot(DPT.objects.DPObject):
    def __init__(self, data, title="Test window", name="", ext="mat"):
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
        return max(0, min(i, self.data.shape[0]-1))

    def plot(self, i, ax=None, overlay=False):
        if ax is None:
            ax = gca()
        if not overlay:
            ax.clear()

        # yet to add argument parsing during object creation and panGUI plot selection
        # need to add if statements for type of plot
        # 
        # Trial - Plot x vs t and y vs t positions per trial
        n = i[0]
        x = self.data.trial_timestamps.to_numpy()
        obj_timestamps = self.data.timestamps.to_numpy()
        print(x)
        print(obj_timestamps)
        trial_start_time = obj_timestamps[x[n][0].astype(int)]
        trial_cue_time = obj_timestamps[x[n][1].astype(int)] - trial_start_time - 1
        trial_end_time = obj_timestamps[x[n][2].astype(int)] - 1

        # timestamps is the x axis to be plotted
        timestamps = obj_timestamps[x[n][0].astype(int) - 501 : x[n][2].astype(int)]
        timestamps = timestamps - trial_start_time
        obj_eye_pos = self.data.eye_pos.to_numpy()
        y = obj_eye_pos[x[n][0].astype(int) - 501 : x[n][2].astype(int)].transpose()
        
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
        timedOut = self.data.timeouts == trial_end_time
        trial_end_time = trial_end_time - trial_start_time
        timedOut = np.nonzero(timedOut.to_numpy)

        if not timedOut: # trial did not timeout
            ax.plot([trial_end_time, trial_end_time], ax.set_ylim(), 'b', LineWidth=0.5)
        else: # trial did timeout
            ax.plot([trial_end_time, trial_end_time], ax.set_ylim(), 'r', LineWidth=0.5)

        ax.set_xlim([-100, trial_end_time + 100]) # set axis boundaries
        ax.set_ylim([0, 1800])
        ax.legend(loc='best')

        # XY - Plots the x and y movement of the eye per trial extract all the trials from one session 
        n = i[0]
        x = self.data.trial_timestamps.to_numpy()
        obj_eye_pos = self.data.eye_pos.to_numpy()
        y = obj_eye_pos[x[n][0].astype(int) : x[n][2].astype(int), :].transpose()
        ax = plotGazeXY(ax, y[0], y[1], 'b') # plot blue circles

        # Calibration - Plot of calibration eye movements
        n = i[0]
        obj_eye_pos = self.data.calib_eye_pos.to_numpy()
        indices = self.data.calib_indices.to_numpy()
        y = obj_eye_pos[indices[n][0].astype(int) : indices[n][2].astype(int), :]
        y = y.transpose()
        ax = plotGazeXY(y[0], y[1], 'b')

        # CalTrial
        n = i[0]
        obj_eye_pos = self.data.calib_eye_pos.to_numpy()
        indices = self.data.calib_indices.to_numpy()
        y = obj_eye_pos[indices[n][0].astype(int) : indices[n][2].astype(int), :]
        ax.plot(y, 'o-', fillstyle='none')
        # lines.Line2D(np.matlib.repmat(obj_eye_pos[indices[n][1].astype(int)], 1, 2), ax.set_ylim())
        ax.set_ylim([0, 1200])
        ax.set_xlim([0, 3500])

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
    # plt.show()
    return ax

def create_obj():
    # create empty object
    el = eyelink()

    # search current directory for .edf files
    files = os.listdir() 
    # for root, dirs, files in os.walk("."): # checks all subfolders as well

    for file in files:
        if file.endswith(Args['FileName']):
            if file.startswith(Args['CalibFileNameChar']):
                # navigation file w/ format: 'Pm_d.edf'
                print('Reading P edf file.\n')
                create_calib_obj(file, el)
            else:
                # calibration file w/ format: 'yymmdd.edf'
                print('Reading day edf file.\n')
                create_nav_obj(file, el)

    return el


######### Calibration file #########
def create_calib_obj(fileName, eyelink):

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
    eye_pos['gx_left'][eye_pos['gx_left'] > Args.get('ScreenX')] = np.nan
    eye_pos['gy_left'][eye_pos['gy_left'] < 0] = np.nan
    eye_pos['gy_left'][eye_pos['gy_left'] > Args.get('ScreenY')] = np.nan
    # print(eye_pos[1:10])
    
    # numSets
    numSets = 1

    eyelink.calib_eye_pos = eye_pos
    eyelink.calib_indices = indices
    eyelink.calib_trial_timestamps = trial_timestamps
    eyelink.calib_numSets = numSets

    return

######### Navigation file #########
def create_nav_obj(fileName, eyelink):

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
    eye_pos['gx_left'][(eye_pos['gx_left'] < 0) | (eye_pos['gx_left'] > Args.get('ScreenX'))] = np.nan
    eye_pos['gy_left'][(eye_pos['gy_left'] < 0) | (eye_pos['gy_left'] > Args.get('ScreenY'))] = np.nan

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

    eyelink.expTime = expTime
    eyelink.timestamps = timestamps
    eyelink.eye_pos = eye_pos
    eyelink.timeouts = timeouts
    eyelink.noOfTrials = noOfTrials
    eyelink.fix_event = fix_event
    eyelink.fix_times = fix_times
    eyelink.sacc_event = sacc_event
    eyelink.numSets = numSets
    eyelink.trial_timestamps = trial_timestamps
    eyelink.trial_codes = trial_codes
    eyelink.session_start = session_start
    eyelink.session_start_index = session_start_index

    return

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