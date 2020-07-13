from pyedfread import edfread
import numpy as np
import numpy.matlib
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import DataProcessingTools as DPT

class Eyelink(DPT.DPObject):
    '''
    Eyelink(redoLevel=0, saveLevel=0)
    '''
    filename = 'eyelink.hkl'
    argsList = [('ObjectLevel', 'Session'), ('FileName', '.edf'), ('CalibFileNameChar', 'P'), 
    ('NavDirName', 'session0'), ('CalibDirName', 'sessioneye'), ('ScreenX', 1920), ('ScreenY', 1080),
    ('NumTrialMessages', 3), ('TriggerMessage', 'Trigger Version 84'), ('StartFromDay', False)]

    def __init__(self, *args, **kwargs):
        level = 'session'
        ll = DPT.levels.level(os.getcwd())
        print('enter', ll)
        # check if StartFromDay is True
        if kwargs.get('StartFromDay'):
            rr = DPT.levels.resolve_level('day', ll)
        else:
            rr = DPT.levels.resolve_level('session', ll)
        
            with DPT.misc.CWD(rr):
                #print('current', DPT.levels.level(os.getcwd()))
                DPT.DPObject.__init__(self, *args, **kwargs)
        print('exit', DPT.levels.level(os.getcwd()))
        
    def create(self, *args, **kwargs):
        # initialize fields in eyelink object
        self.calib_trial_timestamps = pd.DataFrame()
        self.calib_indices = pd.DataFrame()
        self.calib_eye_pos = pd.DataFrame()
        self.calib_numSets = []

        self.trial_timestamps = pd.DataFrame()
        self.eye_pos = pd.DataFrame()
        self.numSets =  []
        self.expTime = []
        self.timestamps = pd.DataFrame()
        self.timeouts = pd.DataFrame()
        self.noOfTrials = []
        self.fix_event = pd.DataFrame()
        self.fix_times = pd.DataFrame()
        self.sacc_event = pd.DataFrame()
        self.trial_codes = pd.DataFrame() # get trial codes from rplparallel instead
        self.session_start = []
        self.session_start_index = []
        
        # cd into day level directory
        ll = DPT.levels.level(os.getcwd())
        rr = DPT.levels.resolve_level('day', ll)

        with DPT.misc.CWD(rr):
            files = os.listdir()
            calib_files = [i for i in files if i.endswith(self.args['FileName']) and self.args['CalibFileNameChar'] in i]
            nav_files = [i for i in files if i.endswith(self.args['FileName']) and self.args['CalibFileNameChar'] not in i]

            for file in calib_files:
                # calibration file w/ format: 'Pm_d.edf'
                print('Reading calibration edf file.\n')

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
                eye_pos = eye_pos[(eye_pos.T != 0).any()]
                # print(eye_pos[1:10])
                
                # numSets
                numSets = 1

                self.calib_eye_pos = eye_pos
                self.calib_indices = indices
                self.calib_trial_timestamps = trial_timestamps
                self.calib_numSets.append(numSets)

                #rr = DPT.levels.resolve_level(self.args['CalibDirName'], ll)
                #with DPT.misc.CWD(rr):
                os.chdir(self.args['CalibDirName'])    
                self.save()
                os.chdir('..')

            for file in nav_files:
                # navigation file w/ format: 'yymmdd.edf'
                print('Reading navigation edf file.\n')

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

                sessionName = []
                dirs = os.listdir()
                for file_name in dirs:
                    if file_name.startswith(self.args['NavDirName']):
                        sessionName.append(file_name)
                actualSessionNo = len(sessionName)

                ###### loop for current session ######
                current_Session = 1

                # eye_positions
                eye_pos = samples[['gx_left', 'gy_left']].copy()
                eye_pos['gx_left'][(eye_pos['gx_left'] < 0) | (eye_pos['gx_left'] > self.args['ScreenX'])] = np.nan
                eye_pos['gy_left'][(eye_pos['gy_left'] < 0) | (eye_pos['gy_left'] > self.args['ScreenY'])] = np.nan
                eye_pos = eye_pos[(eye_pos.T != 0).any()]
                    
                # expTime
                expTime = samples['time'].iloc[0] - 1

                # timestamps
                timestamps = samples['time'].replace(0, np.nan).dropna(axis=0, how='any').fillna(0) - 2198659

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
                trial_codes = trial_codes.astype(np.float64)

                # session_start
                samples2, events2, messages2 = pread(
                    file, trial_marker=b'Trigger Version 84')
                session_start = messages2['trialid_time'].iloc[1]
                
                messages2 = messages2.drop(msg_cols, 1)
                # session_start_index
                session_start_index = session_start - expTime

                # account for multiple sessions
                # save into those directories

                self.expTime.append(expTime)
                self.timestamps = timestamps
                self.eye_pos = eye_pos
                self.timeouts = timeouts
                self.noOfTrials.append(noOfTrials)
                self.fix_event = fix_event
                self.fix_times = fix_times
                self.sacc_event = sacc_event
                self.numSets.append(numSets)
                self.trial_timestamps = trial_timestamps
                self.trial_codes = trial_codes
                self.session_start.append(session_start)
                self.session_start_index.append(session_start_index)
                self.setidx = np.zeros((self.trial_timestamps.shape[0],), dtype=np.int)

                session_dir = self.args['NavDirName'] + str(current_Session)
                # rr = DPT.levels.resolve_level(session_dir, ll)
                # with DPT.misc.CWD(rr):
                os.chdir(session_dir)
                self.save()
                os.chdir('..')

    def append(self, df):
        # update fields in parent
        DPT.DPObject.append(self, df)

        # update fields in child
        self.calib_trial_timestamps = pd.concat(self.calib_trial_timestamps, df.calib_trial_timestamps)
        self.calib_indices = pd.concat(self.calib_indices, df.calib_indices)
        self.calib_eye_pos = pd.concat(self.calib_eye_pos, df.calib_eye_pos)
        self.calib_numSets.append(df.numSets)

        self.trial_timestamps = pd.concat(self.trial_timestamps, df.trial_timestamps)
        self.eye_pos = pd.concat(self.eye_pos, df.eye_pos)
        self.numSets.append(df.numSets)
        self.expTime.append(df.expTime)
        self.timestamps = pd.concat(self.timestamps, df.timestamps)
        self.timeouts = pd.concat(self.timeouts, df.timeouts)
        self.noOfTrials.append(df.noOfTrials)
        self.fix_event = pd.concat(self.fix_event, df.fix_event)
        self.fix_times = pd.concat(self.fix_times, df.fix_times)
        self.sacc_event = pd.concat(self.sacc_event, df.sacc_event)
        self.trial_codes = pd.concat(self.trial_codes, df.trial_codes)
        self.session_start.apppend(df.session_start)
        self.session_start_index.append(df.session_start_index)

    def update_idx(self, i):
        return max(0, min(i, len(self.setidx)-1))

    def plot(self, i=None, getNumEvents=False, getLevels=False, getPlotOpts=False, ax=None, **kwargs):
        # set plot options
        plotopts = {'Plot Options': DPT.objects.ExclusiveOptions(['XT', 'XY', 'FixXY', 'FixXT', 'SaccFix'], 0), "SaccFixSession": False}

        if getPlotOpts:
            return plotopts

        plot_type = plotopts['Plot Options'].selected()
        
        if getNumEvents:
            # Return the number of events avilable
            if plottype == 'SaccFix':
                return 1, 0
        elif plottype == 'SaccFixSession':
            #return number of sessions and which session current trial belongs to
            print('.')
        else:
            if i is not None:
                nidx = i
            else:
                nidx = 0
            return len(self.setidx), nidx

        if getLevels:        
            # Return the possible levels for this object
            return ['session', 'trial', 'all']

        if ax is None:
            ax = plt.gca()
        
        ax.clear()
        
        # Trial - Plot x vs t and y vs t positions per trial
        if (plot_type == 'XT'):
            x = self.trial_timestamps.to_numpy()
            obj_timestamps = self.timestamps.to_numpy()

            trial_start_time = obj_timestamps[x[idx][0].astype(int)]
            trial_cue_time = obj_timestamps[x[idx][1].astype(int)] - trial_start_time - 1
            trial_end_time = obj_timestamps[x[idx][2].astype(int)] - 1

            # timestamps is the x axis to be plotted
            timestamps = obj_timestamps[x[idx][0].astype(int) - 501 : x[idx][2].astype(int)]
            timestamps = timestamps - trial_start_time
            obj_eye_pos = self.eye_pos.to_numpy()
            y = obj_eye_pos[x[idx][0].astype(int) - 501 : x[idx][2].astype(int)].transpose()
            
            # plot x axis data
            ax.plot(timestamps, y[:][0], 'b-', LineWidth=0.5, Label='X position')
            # create title
            # ax.set_title('Eye Movements versus Time for Trial ' + str(i))
            dir = self.dirs[0]
            subject = DPT.levels.get_shortname("subject", dir)
            date = DPT.levels.get_shortname("day", dir)
            session = DPT.levels.get_shortname("session", dir)
            ax.set_title('Eye Movements versus Time - ' + subject + date + session)
            # label axis
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
            # ax.set_ylim([0, 1800])
            ax.legend(loc='best')

        elif (plot_type == 'XY'):
            # XY - Plots the x and y movement of the eye per trial extract all the trials from one session 
            x = self.trial_timestamps.to_numpy()
            obj_eye_pos = self.eye_pos.to_numpy()
            y = obj_eye_pos[x[idx][0].astype(int) : x[idx][2].astype(int), :].transpose()
            ax = plotGazeXY(self, idx, ax, y[0], y[1], 'b') # plot blue circles

        elif (plot_type == 'FixXY'):
            # Calibration - Plot of calibration eye movements
            obj_eye_pos = self.calib_eye_pos.to_numpy()
            indices = self.calib_indices.to_numpy()
            y = obj_eye_pos[indices[idx][0].astype(int) : indices[idx][2].astype(int), :]
            y = y.transpose()
            ax = plotGazeXY(self, idx, ax, y[0], y[1], 'b')

        elif (plot_type == 'FixXT'):
            # Fixation of x vs t
            obj_eye_pos = self.calib_eye_pos.to_numpy()
            indices = self.calib_indices.to_numpy()
            y = obj_eye_pos[indices[idx][0].astype(int) : indices[idx][2].astype(int), :]
            ax.plot(y, 'o-', fillstyle='none')
            lines.Line2D(np.matlib.repmat(obj_eye_pos[indices[idx][1].astype(int)], 1, 2), ax.set_ylim())
            
            dir = self.dirs[0]
            subject = DPT.levels.get_shortname("subject", dir)
            date = DPT.levels.get_shortname("day", dir)
            session = DPT.levels.get_shortname("session", dir)
            ax.set_title(subject + date + session)

        elif (plot_type == 'SaccFix'):
            # SaccFix - Histogram of fixations and saccades per session
            sacc_durations = self.sacc_event.to_numpy()
            fix_durations = self.fix_event.to_numpy()

            sacc_durations = sacc_durations[sacc_durations != 0]
            fix_durations = fix_durations[fix_durations != 0]

            lower = np.amin(fix_durations)
            upper = np.amax(fix_durations)
            
            edges = np.arange(lower, upper, 25).tolist()
            edges = [x for x in edges if x <= 1000]

            ax.hist(sacc_durations, density=False, alpha=0.5, color='#31b4e8', bins=edges, label='Saccades', edgecolor='black', linewidth=0.3)
            ax.hist(fix_durations, density=False, alpha=0.5, color='#ed7f18', bins=edges, label='Fixations', edgecolor='black', linewidth=0.3)
            
            dir = self.dirs[0]
            subject = DPT.levels.get_shortname("subject", dir)
            date = DPT.levels.get_shortname("day", dir)
            session = DPT.levels.get_shortname("session", dir)
            ax.set_title('Distribution of Saccades and Fixations - ' + subject + date + session)
            ax.set_xlabel ('Duration (ms)')
            ax.set_ylabel ('# of events')
            ax.legend(loc='best')
        
        return ax

        
# plotGazeXY helper method to plot gaze position. Uses matlab's plot function
def plotGazeXY(self, i, ax, gx, gy, lineType):
    ax.scatter(gx, gy, color='none', edgecolor=lineType)
    ax.invert_yaxis() # reverse current y axis

    #currentAxis = ax.gca() # draw rect to represent the screen
    ax.add_patch(patches.Rectangle((0, 0), 1920, 1080, fill=None, alpha=1, lw=0.5))

    dir = self.dirs[0]
    subject = DPT.levels.get_shortname("subject", dir)
    date = DPT.levels.get_shortname("day", dir)
    session = DPT.levels.get_shortname("session", dir)
    ax.set_title('Calibration Eye movements - ' + subject + date + session)
    ax.set_xlabel('Gaze Position X (screen pixels)')
    ax.set_ylabel(('Gaze Position Y (screen pixels)'))
    ax.set_xlim([0, 2000]) # set axis boundaries, use max
    ax.set_ylim([1300, 0])

    return ax

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