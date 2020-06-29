from pyedfread import edfread
import numpy as np
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import PanGUI

Args = {'RedoLevels': 0, 'SaveLevels': 0, 'Auto': 0, 'ArgsOnly': 0, 'ObjectLevel': 'Session',
        'FileName': '.edf', 'CalibFileNameChar': 'P', 'EventTypeNum': 24,
        'NavDirName': 'session0*', 'SessionEyeName': 'sessioneye',
        'ScreenX': 1920, 'ScreenY': 1080, 'NumMessagesToClear': 7,
        'TriggerMessage': 'Trigger Version 84', 'NumTrialMessages': 3,
        'StartMessage': 'Start_Trial', 'CueMessage': 'Cue_Offset'}

class eyelink:
    def __init__(self):
        self.nav_trial_timestamps = pd.DataFrame()
        self.nav_indices = pd.DataFrame()
        self.nav_eye_pos = pd.DataFrame()
        self.nav_numSets = 0

        self.trial_timestamps = pd.DataFrame()
        self.indices = pd.DataFrame()
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

######### Navigation file #########
def create_nav_obj(fileName, eyelink):

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
    eye_pos['gy_left'][eye_pos['gy_left'] < 0] = np.nan
    # print(eye_pos[0:10])

    # numSets
    numSets = 1

    eyelink.nav_eye_pos = eye_pos
    eyelink.nav_indices = indices
    eyelink.nav_trial_timestamps = trial_timestamps
    eyelink.nav_numSets = numSets

    return

######### Calibration file #########
def create_calib_obj(fileName, eyelink):

    samples, events, messages = pread(
        fileName, trial_marker=b'Start Trial')

    '''
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
    '''

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
    fix_event = fix_event.loc[events['type']
                              == 'fixation']  # get fixations only
    fix_event = fix_event.iloc[3:]  # 3 might not hold true for all files

    # fix_times
    fix_times = pd.concat([events['start'], events['end'],
                           duration], axis=1, sort=False)
    fix_times = fix_times.loc[events['type']
                              == 'fixation']  # get fixations only
    fix_times = fix_times.iloc[3:]
    # print(fix_times[0:10])

    # sacc_event
    sacc_event = events['end'] - events['start']  # difference is duration
    sacc_event = sacc_event.loc[events['type']
                                == 'saccade']  # get fixations only
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

    # session_start_index
    session_start_index = session_start - expTime

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


def plot(obj):
    n = 0

    # if statements are temporary, must be commented out to run
    # missing argument processing in the shell
    
    if (Args.Trial):
        # Plot x vs t and y vs t positions per trial
        x = obj.trial_timestamps.to_numpy()
        obj_timestamps = obj.timestamps.to_numpy()
        trial_start_time = obj_timestamps[x[n][0].astype(int)]
        trial_cue_time = obj_timestamps[x[n][1].astype(int)] - trial_start_time - 1
        trial_end_time = obj_timestamps[x[n][2].astype(int)] - 1

        # timestamps is the x axis to be plotted
        timestamps = obj_timestamps[x[n][0].astype(int) - 501 : x[n][2].astype(int)]
        timestamps = timestamps - trial_start_time
        obj_eye_pos = obj.eye_pos.to_numpy()
        y = obj_eye_pos[x[n][0].astype(int) - 501 : x[n][2].astype(int)].transpose()
        
        # plot x axis data
        plt.plot(timestamps, y[:][0], 'b-', LineWidth=0.5, Label='X position')
        # label axis
        plt.title('Eye Movements versus Time for Trial -')
        plt.xlabel('Time (ms)')
        plt.ylabel('Position (screen pixels)')
        # plot y axis
        plt.plot(timestamps, y[:][1], 'g-', LineWidth=0.5, Label='Y position')
        
        # Plotting lines to mark the start, cue offset, and end/timeout for the trial
        plt.plot([0, 0], plt.ylim(), 'g', LineWidth=0.5)
        plt.plot([trial_cue_time, trial_cue_time], plt.ylim(), 'm', LineWidth=0.5)
        timedOut = obj.timeouts == trial_end_time
        trial_end_time = trial_end_time - trial_start_time
        timedOut = np.nonzero(timedOut.to_numpy)

        if not timedOut: # trial did not timeout
            plt.plot([trial_end_time, trial_end_time], plt.ylim(), 'b', LineWidth=0.5)
        else: # trial did timeout
            plt.plot([trial_end_time, trial_end_time], plt.ylim(), 'r', LineWidth=0.5)

        plt.xlim([0, trial_end_time]) # set axis boundaries
        plt.ylim([0, 1800])
        plt.show()
 
    elif (Args.XY):
        # Plots the x and y movement of the eye per trial 
        # extract all the trials from one session 
        
        x = obj.trial_timestamps.to_numpy()
        obj_eye_pos = obj.eye_pos.to_numpy()
        y = obj_eye_pos[x[n][0].astype(int) : x[n][2].astype(int), :].transpose()
        plotGazeXY(y[0], y[1], 'b') # plot blue circles
        
    elif (Args.Calibration):
        # working on it
    
    else: # some other argument
        # Histogram of fixations and saccades per session
        sacc_durations = obj.sacc_event.to_numpy()
        fix_durations = obj.fix_event.to_numpy()
        
        sacc_durations = sacc_durations[sacc_durations != 0]
        fix_durations = fix_durations[fix_durations != 0]

        lower = np.amin(fix_durations)
        upper = np.amax(fix_durations)
        
        edges = np.arange(lower, upper, 25).tolist()
        edges = [x for x in edges if x <= 1000]

        plt.hist(sacc_durations, density=1, alpha=0.5, color='#31b4e8', bins=edges, label='Saccades', edgecolor='black', linewidth=0.3)
        plt.hist(fix_durations, density=1, alpha=0.5, color='#ed7f18', bins=edges, label='Fixations', edgecolor='black', linewidth=0.3)
        plt.title ('Distribution of Saccades and Fixations for the Session')
        plt.xlabel ('Duration (ms)')
        plt.ylabel ('# of events')

        plt.xlim([0, 1000]) # set axis boundaries
        plt.ylim([0, 0.04])
        plt.show()
    

# plotGazeXY helper method to plot gaze position. Uses matlab's plot function
def plotGazeXY(gx, gy, lineType):
    plt.scatter(gx, gy, color='none', edgecolor=lineType)
    plt.gca().invert_yaxis() # reverse current y axis
    # rect = patches.Rectangle# draw rect to represent the screen
    # rect = plt.Rectangle((0,0), 1920, 1080, )
    # plt.gca().add_patch(rect)
    plt.title('Calibration Eye movements from session')
    plt.xlabel('Gaze Position X (screen pixels)')
    plt.ylabel(('Gaze Position Y (screen pixels)'))

    plt.xlim([0, 2000]) # set axis boundaries
    plt.ylim([1200, 0])
    plt.show()

def main():
    # create empty h5 file
    hf = h5py.File('eyelink.hdf5', mode='w')

    # create empty object
    el = eyelink()

    # search current directory for .edf files
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(Args['FileName']):
                if file.startswith(Args['CalibFileNameChar']):
                    # navigation file w/ format: 'Pm_d.edf'
                    print('Reading P edf file.\n')
                    create_nav_obj(file, el)

                    hf.create_dataset('nav_numSets', data=el.nav_numSets)
                    hf.create_dataset('nav_trial_timestamps', data=el.nav_trial_timestamps)
                    hf.create_dataset('nav_indices', data=el.nav_indices)
                    hf.create_dataset('nav_eye_pos', data=el.nav_eye_pos)

                else:
                    # calibration file w/ format: 'yymmdd.edf'
                    print('Reading day edf file.\n')
                    create_calib_obj(file, el)

                    hf.create_dataset('numSets', data=el.numSets)
                    hf.create_dataset('trial_timestamps', data=el.trial_timestamps)
                    hf.create_dataset('indices', data=el.indices)
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

    plot(el)


if __name__ == "__main__":
    main()
