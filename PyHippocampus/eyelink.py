from pyedfread import edfread
import numpy as np
import numpy.matlib
import pandas as pd
import hickle as hkl
import h5py
import os
import math
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
import DataProcessingTools as DPT
from .rplparallel import RPLParallel
import time

class Eyelink(DPT.DPObject):
    '''
    Eyelink(redoLevel=0, saveLevel=0)
    '''
    filename = 'eyelink.hkl'
    argsList = [('ObjectLevel', 'Session'), ('FileName', '.edf'), ('CalibFileNameChar', 'P'), 
    ('NavDirName', 'session0'), ('DirName', 'session*'), ('CalibDirName', 'sessioneye'), 
    ('ScreenX', 1920), ('ScreenY', 1080), ('NumTrialMessages', 3), ('TriggerMessage', 'Trigger Version 84'), 
    ('StartFromDay', False), ('RowsToClear', 3), ('TimestampOffset', 2198659)]
    level = 'session'

    def __init__(self, *args, **kwargs):
        ll = DPT.levels.level(os.getcwd())
        
        # check if StartFromDay is True
        if kwargs.get('StartFromDay'):
            rr = DPT.levels.resolve_level('day', ll)
        else:
            rr = DPT.levels.resolve_level('session', ll)
        
            with DPT.misc.CWD(rr):
                DPT.DPObject.__init__(self, *args, **kwargs)
        
    def create(self, *args, **kwargs):
        # initialize fields in eyelink object
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
        self.trial_codes = pd.DataFrame()
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
                trial_timestamps = trial_timestamps.reset_index(drop=True)

                # indices
                index_1 = (messages['trialid_time'] - samples['time'].iloc[0]).iloc[1:]
                index_2 = (time_split['time_0'] - samples['time'].iloc[0]).iloc[1:]
                index_3 = (time_split['time_1'] - samples['time'].iloc[0]).iloc[1:]
                indices = pd.concat([index_1, index_2, index_3], axis=1, sort=False)

                # eye_positions
                eye_pos = samples[['gx_left', 'gy_left']].copy()
                eye_pos['gx_left'][eye_pos['gx_left'] < 0] = np.nan
                eye_pos['gx_left'][eye_pos['gx_left'] > self.args['ScreenX']] = np.nan
                eye_pos['gy_left'][eye_pos['gy_left'] < 0] = np.nan
                eye_pos['gy_left'][eye_pos['gy_left'] > self.args['ScreenY']] = np.nan
                eye_pos = eye_pos[(eye_pos.T != 0).any()]
                
                # numSets
                numSets = 1

                self.eye_pos = eye_pos
                self.indices = indices
                self.trial_timestamps = trial_timestamps
                self.numSets.append(numSets)

                # will implement with processDirs instead
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

                sessionName = []
                dirs = os.listdir()
                for file_name in dirs:
                    if file_name.startswith(self.args['NavDirName']):
                        sessionName.append(file_name)
                actualSessionNo = len(sessionName)

                ###### will set up loop for current session ######

                current_Session = 1

                # eye_positions
                eye_pos = samples[['gx_left', 'gy_left']].copy()
                eye_pos['gx_left'][(eye_pos['gx_left'] < 0) | (eye_pos['gx_left'] > self.args['ScreenX'])] = np.nan
                eye_pos['gy_left'][(eye_pos['gy_left'] < 0) | (eye_pos['gy_left'] > self.args['ScreenY'])] = np.nan
                eye_pos = eye_pos[(eye_pos.T != 0).any()]
                    
                # expTime
                expTime = samples['time'].iloc[0] - 1

                # timestamps
                timestamps = samples['time'].replace(0, np.nan).dropna(axis=0, how='any').fillna(0)

                # timeout
                timeouts = messages['Timeout_time'].dropna()

                # noOfTrials
                noOfTrials = len(messages) - 1

                # fix_event
                duration = events['end'] - events['start']
                fix_event = duration  # difference is duration
                fix_event = fix_event.loc[events['type'] == 'fixation']  # get fixations only
                fix_event = fix_event.iloc[self.args['RowsToClear']:]
                fix_event = fix_event.reset_index(drop=True)

                # fix_times
                fix_times = pd.concat([events['start'], events['end'],
                                    duration], axis=1, sort=False)
                fix_times = fix_times.loc[events['type'] == 'fixation'] # get fixations only
                fix_times['start'] = fix_times['start']
                fix_times['end'] = fix_times['end']
                fix_times = fix_times.iloc[self.args['RowsToClear']:]
                fix_times = fix_times.reset_index(drop=True)

                # numSets
                numSets = 1

                # trial_timestamps
                timestamps_1 = messages['trialid_time'] - expTime
                timestamps_2 = messages['Cue_time'] - expTime
                timestamps_3 = messages['End_time'] - expTime
                trial_timestamps = pd.concat(
                    [timestamps_1, timestamps_2, timestamps_3], axis=1, sort=False)
                trial_timestamps = trial_timestamps.iloc[1:]

                samples2, events2, messages2 = pread(
                    file, trial_marker=b'Trigger Version 84')
                # events2 = events2.drop(event_cols, 1)
                # messages2 = messages2.drop(msg_cols, 1)
                
                # sacc_event 
                sacc_event = pd.DataFrame()
                trigger_m = messages2['trialid_time'].dropna().tolist()
                trigger_m.append(999999999.0)

                for i in range(actualSessionNo):
                    new_event = events[(events['start'] > trigger_m[i]) & (events['start'] < trigger_m[i+1]) & (events['type'] == 'saccade')]
                    duration = (new_event['end'] - new_event['start']).reset_index(drop=True)
                    sacc_event = pd.concat([sacc_event, duration], axis=1)
                sacc_event = sacc_event.fillna(0).astype(int)

                # session_start
                session_start = messages2['trialid_time'].iloc[1]
                
                # session_start_index
                session_start_index = session_start - expTime

                # setup for m dataframe by fetching events messages
                # the two parts below merges dataframe columns of start trial, cue offset and end trial
                # by sorting trialid_time in ascending order
                trigger_ver = messages[['Trigger_time', 'Trigger']]
                trigger_split = trigger_ver['Trigger'].apply(pd.Series) # expand Cue column (a list) into columns of separate dataframe
                trigger_split = trigger_split.rename(columns=lambda x: 'trigger_' + str(x)) # rename columns
                trigger_split['trigger_1'] = 'Trigger Version ' + trigger_split['trigger_1'].astype(str) # append Cue Offset string to each value
                trigger_ver = pd.concat([trigger_ver, trigger_split['trigger_1']], axis=1, sort=False)
                trigger_ver = trigger_ver.drop('Trigger', 1)
                trigger_ver = trigger_ver.rename(columns={'Trigger_time':'trialid_time', 'trigger_1':'trialid '}) # rename columns
                
                start_trial = messages[['trialid_time', 'trialid ']]

                cue_offset = messages[['Cue_time', 'Cue']]
                cue_split = cue_offset['Cue'].apply(pd.Series) 
                cue_split = cue_split.rename(columns=lambda x: 'cue_' + str(x)) 
                cue_split['cue_1'] = 'Cue Offset ' + cue_split['cue_1'].astype(str) 
                cue_offset = pd.concat([cue_offset, cue_split['cue_1']], axis=1, sort=False)
                cue_offset = cue_offset.drop('Cue', 1)
                cue_offset = cue_offset.rename(columns={'Cue_time':'trialid_time', 'cue_1':'trialid '}) 

                end_trial = messages[['End_time', 'End']]
                end_split = end_trial['End'].apply(pd.Series)
                end_split = end_split.rename(columns=lambda x: 'end_' + str(x))
                end_split['end_1'] = 'End Trial ' + end_split['end_1'].astype(str) 
                end_trial = pd.concat([end_trial, end_split['end_1']], axis=1, sort=False)
                end_trial = end_trial.drop('End', 1)
                end_trial = end_trial.rename(columns={'End_time':'trialid_time', 'end_1':'trialid '})

                timeout = messages[['Timeout_time', 'Timeout']].dropna()
                timeout_split = timeout['Timeout'].apply(pd.Series)
                timeout_split = timeout_split.rename(columns=lambda x: 'time_' + str(x))
                timeout_split['time_0'] = 'Timeout ' + timeout_split['time_0'].astype(str)
                timeout = pd.concat([timeout, timeout_split['time_0']], axis=1, sort=False)
                timeout = timeout.drop('Timeout', 1)
                timeout = timeout.rename(columns={'Timeout_time':'trialid_time', 'time_0':'trialid '})

                messageEvent = pd.concat(
                    [trigger_ver, start_trial, cue_offset, end_trial, timeout], axis=0)
                messageEvent = messageEvent.sort_values(by=['trialid_time'], ascending=True) 
                messageEvent = (messageEvent.dropna()).reset_index(drop=True) # drop nans

                m = messageEvent['trialid '].to_numpy()

                s = self.args['TriggerMessage']

                if s != '':
                    sessionIndex = [i for i in range(len(m)) if m[i] == s] # return indices of trigger messages in m
                    noOfSessions = len(messages2.index) - 1 # find length of dataframe

                    extraSessions = 0
                    if noOfSessions > actualSessionNo:
                        print('EDF file has extra sessions!')
                        extraSessions = actualSessionNo - noOfSessions
                    else:
                        print('EDF file has fewer sessions!')

                    #preallocate variables
                    trialTimestamps = np.zeros((m.shape[0], 3*noOfSessions))
                    noOfTrials = np.zeros((1,1))
                    missingData = []
                    sessionFolder = 1

                    # loop to go through all sessions found in edf file
                    # 1) checks if edf file is complete by calling completeData
                    # 2) fills in the trialTimestamps and missingData tables by indexing
                    # with session index (i)

                    for i in range(0, noOfSessions):
                        idx = sessionIndex[i]
                        session = 'session0' + str(i + 1)
                        if i == noOfSessions-1:
                            [corrected_times, tempMissing, flag] = completeData(self, events, samples, m[idx:], messageEvent[idx:], session, extraSessions)
                        else:
                            idx2 = sessionIndex[i+1]
                            [corrected_times, tempMissing, flag] = completeData(self, events, samples, m[idx:idx2], messageEvent[idx:idx2], session, extraSessions)
                        if flag == 0:
                            l = 1 + (sessionFolder-1)*3
                            u = 3 + (sessionFolder-1)*3
                            row = corrected_times.shape[0]
                            trialTimestamps[0:row, l-1:u] = corrected_times
                            noOfTrials[0, sessionFolder-1] = corrected_times.shape[0]
                            missingData = missingData.append(tempMissing)
                            sessionFolder = sessionFolder + 1
                        else:
                            print('Dummy Session skipped', i, '\n')

                # edit the size of the array and remove all zero rows and extra columns
                trialTimestamps = trialTimestamps[~np.all(trialTimestamps == 0, axis=1), :]
                trialTimestamps = trialTimestamps[:, ~np.all(trialTimestamps == 0, axis=0)]
                trialTimestamps = trialTimestamps.astype(int)

                # turn trialTimestamps into dataframe
                trial_timestamps = pd.DataFrame({'Start': trialTimestamps[:, 0], 'Cue': trialTimestamps[:, 1], 'End': trialTimestamps[:, 2]})
                trial_timestamps = trial_timestamps - expTime

                os.chdir('session0' + str(current_Session))

                rpl = hkl.load('rplparallel_b6ee.hkl')
                # print(list(rpl))

                if rpl.get('markers').shape == trial_timestamps.shape:
                    markers = rpl.get('markers')
                    trial_codes = pd.DataFrame(data=markers)
                else:
                    error('markers not consistent')

                os.chdir('..')

                # account for multiple sessions
                # save into those directories
                self.numSets.clear()

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

def completeData(self, events, samples, m, messageEvent, sessionName, moreSessionsFlag):
    # set default flag to 0
    flag = 0
    corrected_times = []
    tempMissing = []

    # the start of the experiment is taken to normalise the data 
    expTime = int(samples['time'].iloc[0] - 1)

    # correct the data for one session
    # create a new matrix that contains all trial messages only
    m = m[m != 'Trigger Version 84']
    messages = m

    # store starting times of all events
    eltimes = messageEvent['trialid_time']

    # read ripple hdf5 file
    os.chdir(sessionName) 

    rpl = hkl.load('rplparallel_b6ee.hkl')
    # print(list(rpl))

    if (rpl.get('numSets') != 0 and (rpl.get('timeStamps')[()]).size != 1):  #no missing rplparallel.mat
        # markers will store all the event numbers in the trial, as taken from the ripple object. 
        # This will be used to mark which events are missing in the eyelink object. 
        # (1-start, 2-cue, 3/4 - end/timeout)
        markers = rpl.get('markers')[()] # get info from rplparallel hdf5
        rpltimeStamps = rpl.get('timeStamps')[()]
        n = len(markers)

        # Check if the rplparallel object is formatted correctly or is missing information
        if n == 1: # if the formatting is 1xSIZE
            df = rpl
            rpl_obj = RPLParallel(Data=True, markers=df.get('markers'), timeStamps=df.get('timeStamps'), rawMarkers=df.get('rawMarkers'), trialIndices=df.get('trialIndices'), sessionStartTime=df.get('sessionStartTime')) 
            
            # how do i know what its called if its randomly generated? there's more than one rplparallel
            
            markers = np.delete(markers, 0)
            rpltimeStamps = np.delete(rpltimeStamps, 0)
            rpltimeStamps = np.delete(rpltimeStamps, rpltimeStamps[np.nonzero(markers == 0)]) # rpltimeStamps(find(~markers)) = [];
            markers = np.delete(markers, np.nonzero(markers == 0))
            n = len(markers) / 3

            if len(markers) % 3 != 0:
                markers = pd.DataFrame(rpl.get('markers'))
                rpltimeStamps = pd.DataFrame(rpl.get('timeStamps'))
                [markers, rpltimeStamps] = callEyelink(self, markers, m, eltimes - expTime, rpltimeStamps)
            else: 
                markers = markers.reshape([3, n])
                rpltimeStamps = rpltimeStamps.reshape([3, n])
                markers = markers.transpose()
                rpltimeStamps = rpltimeStamps.transpose()
            n = markers.shape[0]
            rpl_obj = RPLParallel(Data=True, markers=markers, timeStamps=rpltimeStamps, rawMarkers=df.get('rawMarkers'), trialIndices=df.get('trialIndices'), sessionStartTime=df.get('sessionStartTime'))

        elif n * 3 < m.shape[0]: # If rplparallel obj is missing data, use callEyelink
            if os.path.exists('rplparallel0.hdf5') == False: # use starts with and is file type hkl instead
                df = rpl # extract all fields needed to go into rplparallel constructor
                [markers, rpltimeStamps] = callEyelink(self, markers, m, eltimes-expTime, rpltimeStamps)
                # save object and return
                n = markers.shape[0]
                rpl_obj = RPLParallel(Data=True, markers=markers, timeStamps=rpltimeStamps, rawMarkers=df.get('rawMarkers'), trialIndices=df.get('trialIndices'), sessionStartTime=df.get('sessionStartTime'))
                
        os.chdir('..')

        noOfmessages = messages.shape[0] # stores the number of messages recorded by eyelink in the session

        missing = np.zeros((n, markers.shape[1])) #stores the event that is missing
        rpldurations = np.zeros((n, markers.shape[1])) #stores the rpl durations for filling missing eyelink timestamps
        elTrials = np.zeros((n, markers.shape[1])) #stores the eyelink timestamps for the events in a trial

        # To check if there is more sessions than must be
        if moreSessionsFlag != 0:
            if (n * 3) - messageEvent.shape[0] - 2 >= 100: # use of len of messageEvent, check if the same
                flag = 1
                # create empty dataframes
                elTrials = np.zeros((n, 3)) # pd.DataFrame(columns=range(0,3), index=range(0,n))
                missingData = pd.DataFrame(columns = ['Type', 'Timestamps', 'Messages'], index=range(0,n))
                return
        
        # calculate the durations between rplparallel timestamps 
        # rpldurations(1,1) is assumed to be the time difference between the
        # start of the session and the trial start time
        rpldurations[:, 0] = np.insert(rpltimeStamps[1:, 0] - rpltimeStamps[0:len(rpltimeStamps)-1,2], 0, rpltimeStamps[0,0], axis=0)
        rpldurations[:, 1] = rpltimeStamps[:, 1] - rpltimeStamps[:, 0] # cue time - start time
        rpldurations[:, 2] = rpltimeStamps[:, 2] - rpltimeStamps[:, 1] # end time - cue time

        idx = 1
        n = n * 3 # size of markers
        newMessages = np.zeros((n, 1)) # pd.DataFrame(columns=[0], index=range(0,n)) # stores all the missing messages

        # For loop that goes through the entire rplparallel markers matrix
        # (1) Checks if edf message markers are missing, and accordingly
        # corrects the missing time using the rpldurations
        # (2) ensures that missing trials common to eyelink and rplparallel
        # are deleted
        # (3) creates the missing array that is later addded to the
        # missingData table
        [elTrials, missing, newMessages] = filleye(self, messages, eltimes, rpl)

        # get rid of extra zeros
        [row, _] = np.where(elTrials == 0)
        elTrials = np.delete(elTrials, row, 1) # CHECK: 0 = delete row, 1 = delete column
        n = len(missing.ravel().nonzero())

        if n != 0: # if there are missing messages in this session, make the missingData matrix to add to the .csv file 
            print('Missing messages')
            type = np.empty((n, 1))
            type.fill(24)
            correctedTimes = elTrials[np.where(missing != float(0))[0][0], np.where(missing != float(0))[1][0]]
            newMessages = newMessages.dropna()
            missingData = pd.DataFrame()
            missingData['Type'] = type[0]
            missingData['Timestamps'] = correctedTimes
            missingData['Messages'] = newMessages
        else: #if there are not any missing messages, make the missingData matrix empty 
            print('No missing messages')
            missingData = pd.DataFrame(columns = ['Type', 'Timestamps', 'Messages'], index=range(0,n))
        '''
        # To ensure that the correction of the eyelink object went correctly, 
        # we now plot a histogram of the discrepancies in the start-cue,
        # cue-end and end-start durations for ripple and eyelink objects for
        # the same session to ensure that the data recorded is consistent 
        elTrials = elTrials - expTime
        eldurations[:, 0] = np.insert(elTrials[1:, 0] - elTrials[0:len(elTrials)-1,2], 0, 0, axis=0)
        eldurations[:, 1] = elTrials[:, 1] - elTrials[:, 0] # cue time - start time
        eldurations[:, 2] = elTrials[:, 2] - elTrials[:, 1] # end - cue time
        eldurations = eldurations / 1000 #conversion to seconds
        discrepancies = abs(rpldurations - eldurations) # stores the discrepancy between the two times in seconds
        discrepancies[0, 0] = 0

        # plot the distributions of the durations in ms
        plt.hist(discrepancies, bins=50)
        plt.title('Histogram for rplparallel and eyelink durations')
        plt.xlabel('s')
        plt.ylabel('occurence')
        plt.show()
        '''
    else: #missing rplparallel  
        # assume that there are no missing messages
        os.chdir('..')
        print('Empty object. Just fill up time array\n')
        n = messages.shape[0]
        elTrials = np.zeros((n, 3)) # stores the eyelink timestamps for the events in a trial
        missing = np.zeros((n, 3))

        for i in range(n):
            r = math.floor(i / 3) + 1 # row
            c = i % 3

            if c == 0:
                r = r - 1
                c = 3
            
            missing[r, c] = 0
            if 'Start Trial' in messages[i, 0]:
                elTrials[r, 0] = eltimes[i, 0] - expTime
            elif 'End Trial' in messages[i, 0]:
                elTrials[r, 2] = eltimes[i, 0] - expTime
            elif 'Timeout' in messages[i, 0]:
                elTrials[r, 2] = eltimes[i, 0] - expTime
            elif 'Cue Offset' in messages[i, 0]:
                elTrials[r, 1] = eltimes[i, 0] - expTime

        print('No missing messages')
        missingData = pd.DataFrame(columns = ['Type', 'Timestamps', 'Messages'], index=range(0,n))

    return elTrials, missingData, flag

def filleye(self, messages, eltimes, rpl):
    eyelink_raw = np.empty((1,len(messages)))
    eyelink_raw[:] = np.nan

    for i in range(len(messages)):
        full_text = messages[i]
        full_text = full_text.split() # splits string into list
        full_text = full_text[len(full_text)-1]
        eyelink_raw[0, i] = float(full_text)

    eye_timestamps = eltimes.to_numpy().transpose()
    truth_timestamps = rpl.get('timeStamps')[()]
    truth_timestamps = truth_timestamps * 1000
    truth = rpl.get('markers')[()]

    # data transformed into format used by function
    split_by_ones = np.empty((2000,10))
    split_by_ones[:] = np.nan
    row, col, max_col, start = 1, 1, 1, 1

    for i in range(eyelink_raw.shape[1]): # naively splits the sequence by plausible triples by cue onset
        if (eyelink_raw[0, i] < 20 and eyelink_raw[0, i] > 9) or col > 3:
            row = row + 1
            if col > max_col:
                max_col = col
            col = 1
        elif col != 1:
            if base < 20:
                if eyelink_raw[0, i] != base + 10:
                    if eyelink_raw[0, i] != base + 20:
                        if eyelink_raw[0, i] != base + 30:
                            row = row + 1
                            col = 1
            elif base < 30:
                if eyelink_raw[0, i] != base + 10:
                    if eyelink_raw[0, i] != base + 20:
                        row = row + 1
                        col = 1
            else:
                row = row + 1
                col = 1

        split_by_ones[row, col] = eyelink_raw[0, i]
        base = eyelink_raw[0, i]
        col = col + 1
        if (start == 1):
            start = 0

    if np.sum(~np.isnan(split_by_ones[0,:])) != 0:
        split_by_ones = split_by_ones[1:row+1, 1:max_col] # test out: do I need -1 ?
    else:
        split_by_ones = split_by_ones[2:row+1, 1:max_col]
    
    arranged_array = np.empty((split_by_ones.shape))
    arranged_array[:] = np.nan
    
    for row in range(split_by_ones.shape[0]):
        for col in range(3):
            if np.isnan(split_by_ones[row, col]):
                break
            if split_by_ones[row, col] < 20:
                arranged_array[row, 0] = split_by_ones[row, col]
            elif split_by_ones[row, col] < 30:
                arranged_array[row, 1] = split_by_ones[row, col]
            else:
                arranged_array[row, 2] = split_by_ones[row, col]

    missing_rows = len(truth) - len(arranged_array)

    slice_after = np.empty((missing_rows, 2)) # test - this section accounts for triples that look ok, but are made of two trials with the same posters
    slice_after[:] = np.nan
    slice_index = 1

    for row in range(arranged_array.shape[0]):
        #if (row > 316):
        #    print('debugger')
        if ~np.isnan(arranged_array[row, 0]):
            if ~np.isnan(arranged_array[row, 1]):
                tmp = arranged_array.transpose()
                tmp = tmp.flatten('F')
                idx = np.sum(~np.isnan(tmp[0:3*(row)+1]))
                td = eye_timestamps[idx+1] - eye_timestamps[idx]

                rpl_chunk = truth_timestamps[row:min(max(row, row+missing_rows+1), truth_timestamps.shape[0]), 0:2]
                rpl_chunk_flag = truth[row:min(max(row, row+missing_rows+1), truth_timestamps.shape[0]), 0:2]
                
                rpl_chunk = rpl_chunk[rpl_chunk_flag[:, 0] == arranged_array[row, 0], :]
                rpl_td = rpl_chunk[:, 1] - rpl_chunk[:, 0]

                if np.min(abs(rpl_td - td)) > 1500:
                    slice_after[slice_index, :] = [row, 0]
                    slice_index = slice_index + 1

            elif ~np.isnan(arranged_array[row, 2]):
                tmp = arranged_array.transpose()
                tmp = tmp.flatten('F')
                idx = np.sum(~np.isnan(tmp[0:3*(row)+1]))
                idx3 = np.sum(~np.isnan(tmp[0:3*(row)+3]))
                td = eye_timestamps[idx3] - eye_timestamps[idx]

                rpl_chunk = truth_timestamps[row:min(max(row, row+missing_rows+1), truth_timestamps.shape[0]), 0:3]
                rpl_chunk_flag = truth[row:min(max(row, row+missing_rows+1), truth_timestamps.shape[0]), 0:3]

                rpl_chunk = rpl_chunk[rpl_chunk_flag[:,0] == arranged_array[row,0], :]
                rpl_td = rpl_chunk[:,2] - rpl_chunk[:,0]

                if np.min(abs(rpl_td - td)) > 1500:
                    slice_after[slice_index, :] = [row, 0]
                    slice_index = slice_index + 1

        elif ~np.isnan(arranged_array[row, 1]):
            if ~np.isnan(arranged_array[row, 2]):
                tmp = arranged_array.transpose()
                tmp = tmp.flatten('F')
                idx = np.sum(~np.isnan(tmp[0:3*(row)+2]))
                td = eye_timestamps[idx+1] - eye_timestamps[idx]

                rpl_chunk = truth_timestamps[row:min(max(row, row+missing_rows+1), truth_timestamps.shape[0]), 1:3]
                rpl_chunk_flag = truth[row:min(max(row, row+missing_rows+1), truth_timestamps.shape[0]), 1:3]
                
                rpl_chunk = rpl_chunk[rpl_chunk_flag[:,1] == arranged_array[row,1], :]
                rpl_td = rpl_chunk[:,1] - rpl_chunk[:,0]

                if np.min(abs(rpl_td - td)) > 1500:
                    slice_after[slice_index, :] = [row, 0]
                    slice_index = slice_index + 1

    slice_after = slice_after[0:slice_index-1,:]
    empty_missing_rows = np.empty((missing_rows, 3))
    empty_missing_rows[:] = np.nan
    arranged_array = np.vstack((arranged_array, empty_missing_rows))

    if len(slice_after) != 0:
        for slice in range(len(slice_after), -1, -1): # slices according to previously identified segments
            new_array = np.empty(np.shape(arranged_array)) # can I use len or shape
            new_array[0:slice_after[slice, 0]-1, :] = arranged_array[0:slice_after[slice, 0]-1, :] # is it [0:slice_after[slice,0]-2,:] , variables start at 0
            new_array[slice_after[slice, 0]-1, 0:slice_after[slice, 1]] = arranged_array[slice_after[slice, 0], 0:slice_after[slice,1]] 
            new_array[slice_after[slice, 0], slice_after[slice, 1]:3] = arranged_array[slice_after[slice, 0]-1,slice_after[slice,1]:3]
            arranged_arrya[slice_after[slice, 0]:arranged_array.shape[0], :]
            new_array[slice_after[slice, 0]+1:, :] = arranged_array[slice_after[slice,0]:arranged_array.shape[0]-1,:]    
            arranged_array = new_array
            missing_rows = missing_rows - 1

    for row in range(missing_rows): #this segment attempts to identify where entire trials may have gone missing, by comparing with rpl timings
        error = np.nansum(truth - arranged_array, axis=1) #nansum(dim=2)
        error_index = min(error.ravel().nonzero()) # min(find(error~=0))
        if np.sum(abs(error)) == 0:
            break
        if error_index == 1:
            empty_nan = np.empty((1, 3))
            empty_nan[:] = np.nan
            arranged_array = np.vstack((empty_nan, arranged_array[0:len(arranged_array)-1, :]))

        else:
            for col in range(3):
                if ~np.isnan(arranged_array[error_index-1, col]):
                    pre_id = arranged_array[error_index-1, col] % 10
                    break
                # identity of preceeding trial determined
            # looking up how many trials before this have the same identity
            count = 0
            while (1):
                if error_index-1-count == 0:
                    break
                for col2 in range(3):
                    if ~np.isnan(arranged_array[error_index-1-count, col2]):
                        pre_id_check = arranged_array[error_index-1-count, col2] % 10
                        break
                if pre_id_check != pre_id:
                    break
                if error_index-2-count == 0:
                    break
                count = count + 1
            # count now stores the number of repeated posters before the
            # misalignment has been detected (need to test all possible
            # locations).
            print(count)

            eye_start_trials = np.empty((count+2, 0))
            eye_start_trials[:] = np.nan
            eye_start_count = 1

            esi = 0
            for r in range(arranged_array.shape[0]):
                for c in range(3):
                    if ~np.isnan(arranged_array[r, c]):
                        esi = esi + 1
                    if r >= error_index-count-1 and r <= error_index:
                        if c == 1:
                            if ~np.isnan(arranged_array[r, c]):
                                eye_start_trials[eye_start_count, 0] = eye_timestamps[esi]
                            elif ~np.isnan(arranged_array[r, c+1]):
                                print('taking cue offset and cutting 2seconds to estimate start trial timing')
                                eye_start_trials[eye_start_count, 0] = eye_timestamps[esi+1]-2000
                            else:
                                print('taking end trial and cutting 10seconds to estimate start trial timing')
                                eye_start_trials[eye_start_count, 0] = eye_timestamps[esi+1]-10000
                            eye_start_count = eye_start_count + 1
                rpl_start_trials = truth_timestamps[error_index-count-1:error_index, 0]
                diff_eye = diff[eye_start_trials]
                diff_rpl = diff[rpl_start_trials]
                discrepancy = diff_eye - diff_rpl
                [_,row_to_insert] = max(discrepancy)

                empty_nans = np.zeros((1,3))
                empty_nans[:] = np.nan
                arranged_array = np.concatenate((arranged_array[0:error_index-count-2+row_to_insert, :], empty_nans, arranged_array[error_index-count-1+row_to_insert:, :]), axis=0)
                arranged_array = arranged_array[0:len(arranged_array)-1, :]
    
    if np.nansum(abs(arranged_array.astype(float) - truth.astype(float))) > 0:
        raise ValueError('eyelink was not properly arranged. current arrangement still clashes with ripple')

    missing = truth*np.isnan(arranged_array).astype(float)

    newMessages = pd.DataFrame(index=range(3*truth.shape[0]), columns=range(1))
    flat_truth = truth.transpose()
    flat_truth = flat_truth.flatten('F')
    flat_truth_time = truth_timestamps.transpose()
    flat_truth_time = flat_truth_time.flatten('F')
    flat_eye = arranged_array.transpose()
    flat_eye = flat_eye.flatten('F')
    flat_truth = flat_truth*np.isnan(flat_eye).astype(float)

    for i in range(len(flat_truth)):
        if flat_truth[i] != 0:
            if flat_truth[i] < 20:
                text = 'Start Trial ' + str(flat_truth[i].astype(int))
                newMessages.loc[i] = text
            elif flat_truth[i] < 30:
                text = 'Cue Offset ' + str(flat_truth[i].astype(int))
                newMessages.loc[i] = text
            elif flat_truth[i] < 40:
                text = 'End Trial ' + str(flat_truth[i].astype(int))
                newMessages.loc[i] = text
            else:
                text = 'Timeout ' + str(flat_truth[i].astype(int))
                newMessages.loc[i] = text
    # ready for output

    elTrials = np.zeros((1, 3*missing.shape[0])) # change to empty list
    counter = 1
    eltimes = eltimes.to_list()

    for i in range(len(flat_eye)):
        if ~np.isnan(flat_eye[i]):
            elTrials[0][i] = eltimes[counter]
            counter = counter + 1

    elTrials = elTrials.astype(int)

    for i in range(len(elTrials[0])):
        if elTrials[0][i] == 0:
            if i == 1:
                inv_delta = flat_truth_time[i+1] - flat_truth_time[i]
                elTrials[0][i] = (elTrials[0][i+1] - inv_delta).round()
                print('shouldnt see nans here')
            else:
                delta = flat_truth_time[i] - flat_truth_time[i-1]
                elTrials[0][i] = (elTrials[0][i-1] + delta).round()
                print('shouldnt see nans here')

    elTrials = elTrials.reshape([len(elTrials[0])//3, 3]) # ready for output

    return elTrials, missing, newMessages

def callEyelink(self, markersRaw, messages, eltimes, rpltimeStamps):
    # stores timestamps from eyelink
    eldurations = np.insert(eltimes[1:] - eltimes[0:len(elTrials)-1], 0, 0, axis=0)
    eldurations = eldurations / 1000
    eltimes = eldurations

    # Get rid of the 0s that separate the markers and the first 84 markers 
    # Do the same for the rplparallel timestamps to reconstruct them.
    if (markersRaw[0] == 84):
        markersRaw = markersRaw.pop() # remove first element from list
        rpltimeStamps = rpltimeStamps.pop()
    rpltimeStamps = np.delete(rpltimeStamps, rpltimeStamps[np.nonzero(markers == 0)])
    markers = np.delete(markers, np.nonzero(markers == 0))
    n = markersRaw.shape[1]

    if (n < messages.shape[0]): # messages = m # should be len(messages)
        m = messages.shape[0] # len(messages)

        # first check if edf file is missing something too
        remainder = m % 3
        if (remainder != 0):
            print('Edf file incomplete\n')
        
        markersNew = np.zeros((m, 1))
        timesNew = np.zeros((m, 3))
        print(markersNew)
        print(timesNew)
        idx = 1
        idx2 = 1
        sz = m + n
        count = 1

        for i in range(sz):
            # convert to row-column indexing for markersNew
            r = math.floor(i / 3) + 1 # row
            c = i % 3 # column
            
            if (c == 0):
                r = r - 1
                c = 3

            if (idx2 <= m): #prevent accessing more than the array size 
                # Convert the string to a number for easier manipulation
                # message = ''.join(str(e) for e in messages[idx2][0]) # convert list to number
                message = int(messages[idx2, 0])
            
            if ((math.floor(message / 10) == c) or (math.floor(message / 10) == 4) and c == 3): #ensures that messages itself isn't missing a trial
                if (idx <= n and message == markersRaw[0, idx]): # if the marker exists in rplparallel
                    markersNew[r, c] = markersRaw[0, idx]
                    timesNew[r, c] = rpltimeStamps[0, idx]
                    idx = idx + 1
                else: # rplparallel is missing a marker
                    print('Missing in rplparallel but found in messages\n')
                    if (c == 1 and r != 1):
                        timesNew[r, c] = timesNew[r-1, 2] + eltimes[idx2, 0]
                    elif (c == 1 and r == 1):
                        timesNew[r, c] = rpltimeStamps[0, idx] - eltimes[idx2+1, 0] # maybe dont need +1 in python
                    else:
                        timesNew[r, c] = timesNew[r, c-1] + eltimes[idx2, 0]
                idx2 = idx2 + 1
            else: #check if markersRaw has it instead 
                if ((math.floor(markersRaw[0, idx] / 10) == 3) or (math.floor(markersRaw[0, idx] / 10) == 4 and c == 3)):
                    if (c != 1 and ((markersRaw[0, idx] % 10) == (markersRaw[0, idx - 1] % 10))):
                        markersNew[r, c] = markersRaw[0, idx]
                        timesNew[r, c] = rpltimeStamps[0, idx]
                        print('Missing Data from messages. but found in rplparallel\n')
                        disp([r, c])
                else:
                    markersNew[r, c] = 0
                    timesNew[r, c] = 0
                    print('Use unitymaze\n')
                    count = count + 1
                idx = idx+1
            
            # once done checking
            if (idx > n and idx2 > m):
                break;
    else:
        m = messages.shape[0]
        markersNew = np.zeros((n+m, 3))
        timesNew = np.zeros((n+m, 3))
        idx = 1 # parsing through markers
        idx2 = 1 # parsing through messages
        count = 0
        sz = n + m

        for i in range(sz):
            # index for markersNew
            r = math.floor(i / 3) + 1 # row
            c = i % 3 # column
            
            if (c == 0):
                r = r - 1
                c = 3

            if (idx2 <= m): 
                message = int(messages[idx2, 0])
            
            if (math.floor(markersRaw[0, idx] / 10) == c): #if it is going in the correct column
                if (idx2 <= m and message == markersRaw[0, idx]): #if messages has it we can move ahead
                    idx2 = idx2 + 1
                else:
                    count = count + 1
                markersNew[r, c] = markersRaw[0, idx]
                timesNew[r, c] = rpltimeStamps[0, idx]
                idx = idx + 1
            elif ((math.floor(markersRaw[0, idx] / 10) == 4) and c == 3): #timeout condition
                if (idx2 <= m and message == markersRaw[0, idx]):
                    idx2 = idx2 + 1      
                markersNew[r, c] = markersRaw[0, idx]
                timesNew[r, c] = rpltimeStamps[0, idx]
                idx = idx + 1
            else: # it is missing from rplparallel
                # check if messages has it
                if ((math.floor(message / 10) == c) or (math.floor(message / 10) == 4) and c == 3): # message has it
                    print('Missing in rplparallel. But found in messages\n')
                    markersNew[r, c] = message
                    if (c == 1):
                        timesNew[r, c] = timesNew[r-1, 2] + eltimes[idx2, 0]
                    else:
                        timesNew[r, c] = timesNew[r, c-1] + eltimes[idx2, 0]
                    idx2 = idx2 + 1
                else: # even messages doesnt have it
                    print('Delete Trial\n')
                    markersNew[r, c] = 0
                    timesNew[r, c] = 0
                    count = count + 1
            # to ensure that we can break from the loop and don't need to waste execution time 
            if (idx > n and idx > m):
                break
    
    markersNew = markersNew[np.any(markersNew, 2)][:]
    timesNew = timesNew[np.any(timesNew, 2)][:]
    print(count)

    return markersNew, timesNew