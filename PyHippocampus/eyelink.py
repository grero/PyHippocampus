# To set up:
# pip install pandas
# pip install Cython
# run 'python setup.py install' in pyedfread-master


from pyedfread import edfread
import numpy as np
import pandas as pd
import h5py
import os

Args = {'RedoLevels': 0, 'SaveLevels': 0, 'Auto': 0, 'ArgsOnly': 0, 'ObjectLevel': 'Session',
        'FileName': '.edf', 'CalibFileNameChar': 'P', 'EventTypeNum': 24,
        'NavDirName': 'session0*', 'SessionEyeName': 'sessioneye',
        'ScreenX': 1920, 'ScreenY': 1080, 'NumMessagesToClear': 7,
        'TriggerMessage': 'Trigger Version 84', 'NumTrialMessages': 3,
        'StartMessage': 'Start_Trial', 'CueMessage': 'Cue_Offset'}


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


def create_nav_obj(fileName):

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
    # print(trial_timestamps2)

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
    # print(eye_pos2[0:10])

    # numSets
    numSets = 1

    # add to hdf file

    return

######### Calibration file #########


def create_calib_obj(fileName):

    samples, events, messages = pread(
        fileName, trial_marker=b'Start Trial')

    '''
    Used to filter out unneeded columns of dataframes returned by pread.
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
    # print(expTime)

    # timestamps
    timestamps = samples['time']
    # print(timestamps.iloc[0:10])

    # eye_positions
    eye_pos = samples[['gx_left', 'gy_left']].copy()
    eye_pos['gx_left'][eye_pos['gx_left'] < 0] = np.nan
    eye_pos['gy_left'][eye_pos['gy_left'] < 0] = np.nan
    # print(eye_pos[0:10])

    # timeout
    timeouts = messages['Timeout_time'].dropna()
    # print(timeouts)

    # noOfTrials
    noOfTrials = len(messages) - 1
    # print(noOfTrials)

    # fix_event
    duration = events['end'] - events['start']
    fix_event = duration  # difference is duration
    fix_event = fix_event.loc[events['type']
                              == 'fixation']  # get fixations only
    fix_event = fix_event.iloc[3:]  # 3 might not hold true for all files
    # print(fix_event[0:20])

    # fix_times , start and end columns are not adjusted
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
    # print(sacc_event[0:10])

    # numSets
    numSets = 1

    # trial_timestamps
    timestamps_1 = messages['trialid_time'] - expTime
    timestamps_2 = messages['Cue_time'] - expTime
    timestamps_3 = messages['End_time'] - expTime
    trial_timestamps = pd.concat(
        [timestamps_1, timestamps_2, timestamps_3], axis=1, sort=False)
    trial_timestamps = trial_timestamps[1:]
    # print(trial_timestamps)

    # trial_codes
    trial_id = messages['trialid '].str.replace(
        r'\D', '')  # remove 'Start Trial '
    cue_split = messages['Cue'].apply(pd.Series)
    cue_split = cue_split.rename(columns=lambda x: 'cue_' + str(x))
    end_split = messages['End'].apply(pd.Series)
    end_split = end_split.rename(columns=lambda x: 'end_' + str(x))
    trial_code = pd.concat(
        [trial_id, cue_split['cue_1'], end_split['end_1']], axis=1, sort=False)
    trial_code = trial_code.iloc[1:]
    # print(trial_code[0:10])

    # session_start
    samples2, events2, messages2 = pread(
        fileName, trial_marker=b'Trigger Version 84')
    session_start = messages2['trialid_time'].iloc[1]
    # print(session_start)

    # session_start_index
    session_start_index = session_start - expTime
    # print(session_start_index)

    # write dataframes to hdf file

    return


def create_object():
    # search current directory for .edf files
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(Args['FileName']):
                if file.startswith(Args['CalibFileNameChar']):
                    # navigation file w/ format: 'Pm_d.edf'
                    print('Reading P edf file.\n')
                    create_nav_obj(file)
                else:
                    # calibration file w/ format: 'yymmdd.edf'
                    print('Reading day edf file.\n')
                    create_calib_obj(file)

    return


def main():
    create_object()


if __name__ == "__main__":
    main()
