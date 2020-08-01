import h5py
import time
import numpy as np
import numpy.matlib
import hickle as hkl
import DataProcessingTools as DPT
from .rplparallel import RPLParallel
from .eyelink import Eyelink
from .unity import Unity



def aligning_objects():
    threshold = 0.02

    uf = Unity()
    rp = RPLParallel()
    el = Eyelink()
    el.session_start[0] = el.session_start[0]*el.samplingRate
    el.timestamps = el.timestamps*el.samplingRate
    el.trial_timestamps = el.trial_timestamps*el.samplingRate
    el.fix_times = el.fix_times*el.samplingRate
    el.timestamps = el.timestamps[el.timestamps > 0]
            
    true_timestamps = np.array(rp.timeStamps) * 1000
    a = np.shape(true_timestamps)
    true_timestamps = np.reshape(true_timestamps,a[0]*a[1],order='C')
    
    el_trial_timestamps_flat = np.array(el.trial_timestamps)
    b = np.shape(el_trial_timestamps_flat)
    el_trial_timestamps_flat = np.reshape(el_trial_timestamps_flat,b[0]*b[1],order='C')
    
    uf_unityTriggers_flat = uf.unityTriggers[0]
    c = np.shape(uf_unityTriggers_flat)
    uf_unityTriggers_flat = np.reshape(uf_unityTriggers_flat,c[0]*c[1],order='C')
    
    dubious_counter = 0
    dubious_collector = []  
    
    saving_closest_fix_times = np.array(el.fix_times)
    saving_closest_fix_times = saving_closest_fix_times[:,0:2]
    saving_closest_fix_times = np.transpose(saving_closest_fix_times)
    saving_closest_fix_times = np.reshape(saving_closest_fix_times,(np.shape(saving_closest_fix_times)[0]*np.shape(saving_closest_fix_times)[1]),order='F')
    
    ts = np.array(el.timestamps)
    t = time.time()
    
    difference = float('NaN')
    index = 0    
    for stamps in range(np.shape(ts)[0]):
        if np.isnan(difference):
            difference = ts[stamps] - saving_closest_fix_times[index]    
        else:
            if ts[stamps] - saving_closest_fix_times[index] > 0 :
                if abs(difference) > (abs(ts[stamps]) - saving_closest_fix_times[index]):
                    saving_closest_fix_times[index] = stamps
                else:
                    saving_closest_fix_times[index] = stamps - 1
                difference = float('NaN')
                index = index + 1
            else:
                difference = ts[stamps] - saving_closest_fix_times[index]
        if index >= np.shape(saving_closest_fix_times)[0]:
            break
    
    saving_closest_fix_times = np.reshape(saving_closest_fix_times,(int(np.shape(saving_closest_fix_times)[0]/2),2))
    saving_closest_fix_times += 1
    elapsed = time.time() - t
    
    for j in range(np.shape(true_timestamps)[0]-1):
        
        true_start = true_timestamps[j]
        true_end = true_timestamps[j+1]
        true_diff = true_end - true_start

        current_start = el_trial_timestamps_flat[j]
        current_end = el_trial_timestamps_flat[j+1]
        current_chunk = np.array(el.timestamps)
        current_chunk = current_chunk[int(current_start)-1:int(current_end)]
        current_chunk = current_chunk.astype(float)
        current_diff = current_chunk[np.shape(np.array(current_chunk))[0]-1] - current_chunk[0]
        
        
        current_start_time = current_chunk[0]
        current_end_time = current_chunk[np.shape(current_chunk)[0]-1]
        current_chunk = (current_chunk - current_start_time)* true_diff/current_diff 
        current_chunk = current_chunk + current_start_time
        shifting_needed = current_chunk[np.shape(current_chunk)[0]-1] - current_end_time
    
        el.timestamps[int(current_start)-1:int(current_end)] = np.uint32(current_chunk)
        el.timestamps[int(current_end):np.shape(el.timestamps)[0]] = el.timestamps[int(current_end):np.shape(el.timestamps)[0]]+shifting_needed      
    
        true_diff = true_diff/1000
        
        current_start = uf_unityTriggers_flat[j]+2
        current_end = uf_unityTriggers_flat[j+1]+2
        current_chunk = uf.unityTime[0][current_start-1:current_end]
        current_diff = current_chunk[np.shape(current_chunk)[0]-1] - current_chunk[0]
        current_start_time = current_chunk[0]
        current_end_time = current_chunk[np.shape(current_chunk)[0]-1]

        
        dubious = 0
        if j % 3 == 0:
            discrep = current_diff - true_diff
            #print(j/3)
            #print(discrep)
        elif j % 3 == 1:
            discrep = discrep + current_diff - true_diff;
            #print(discrep)
        else:
            if abs(discrep) > threshold:
                dubious = 1       
        
        current_chunk = (current_chunk - current_start_time) * true_diff/current_diff  
        current_chunk = current_chunk + current_start_time 
        shifting_needed = current_chunk[np.shape(current_chunk)[0]-1] - current_end_time
        
    
        uf.unityTime[0][int(current_start)-1:int(current_end)] = current_chunk
        uf.unityTime[0][int(current_end):np.shape(uf.unityTime[0])[0]] = uf.unityTime[0][int(current_end):np.shape(uf.unityTime[0])[0]] + shifting_needed        
        
        
        if dubious == 1:
            prev_prev_start = uf_unityTriggers_flat[j-2]+1
            chunk_size = np.shape(uf.unityTime[0][prev_prev_start:current_start])[0]
            uf.unityTime[0][prev_prev_start:current_start] = numpy.matlib.repmat(uf.unityTime[0][prev_prev_start],1,chunk_size)
            
            dubious_counter = dubious_counter + 1
            dubious_collector.append(j)
            print('but disparity between rpl and unity was quite large')
            print(discrep)
   
    
    print('dubious counter: ' + str(dubious_counter))
    print('dubious location(s): ' + str(dubious_collector))
    
    markers = np.array(rp.rawMarkers)
    
    if markers[0] == 84: 
        true_session_start = np.array(rp.sessionStartTime)
        #print(true_session_start)
        session_trial_duration = rp.timeStamps[0][0] - true_session_start
        session_trial_duration = session_trial_duration * 1000
        #print(session_trial_duration)
        finding_index = 0
        for i in range(np.shape(el.timestamps)[0]):
            if el.timestamps[i] != el.session_start[0]:
                finding_index += 1
            else:
                break
        #print(finding_index)
        el_session_trial_chunk = np.array(el.timestamps)[finding_index:int(np.array(el.trial_timestamps)[0][0])]
        el_session_trial_chunk.astype(float)
        last_point = el_session_trial_chunk[np.shape(el_session_trial_chunk)[0]-1]
        first_point = el_session_trial_chunk[0]   
        scaled_chunk = ((el_session_trial_chunk - el_session_trial_chunk[0]) / float(last_point - first_point)) * session_trial_duration
        scaled_chunk = scaled_chunk + first_point
        shifting_needed = scaled_chunk[np.shape(scaled_chunk)[0]-1] - last_point
        start = int(np.array(el.trial_timestamps)[0][0])
        end = np.shape(el.timestamps)[0]
        el.timestamps[start:end] = el.timestamps[start:end] + shifting_needed
        el.timestamps[finding_index:start] = scaled_chunk     
        
        target = true_session_start * 1000
        full_shift = np.array(el.session_start) - target
        el.timestamps[:] = np.uint32(el.timestamps[:] - full_shift)
    
        
        working_copy = np.array(el.fix_times)[:,0:2]
        working_copy = np.uint32(working_copy)
        
        TS = np.array(el.timestamps)
        #TS = np.transpose(TS)
        for row in range(np.shape(el.fix_times)[0]):
            for col in range(0,2):
                working_copy[row, col] = TS[int(saving_closest_fix_times[row, col])]
                if col == 0:
                    
                    el.fix_times['start'][row] = working_copy[row, col]
                else:
                    el.fix_times['end'][row] = working_copy[row, col]
        
        
        session_trial_duration = rp.timeStamps[0][0] - true_session_start
        uf_session_trial_chunk = uf.unityTime[0][0:uf.unityTriggers[0][0][0]+2]
        last_point = uf_session_trial_chunk[np.shape(uf_session_trial_chunk)[0]-1]
        scaled_chunk = (uf_session_trial_chunk/last_point) * session_trial_duration
        shifting_needed = scaled_chunk[np.shape(scaled_chunk)[0]-1] - last_point
        
        uf.unityTime[0][uf.unityTriggers[0][0][0]+1:np.shape(uf.unityTime[0])[0]] += shifting_needed
        uf.unityTime[0][0:uf.unityTriggers[0][0][0]+2] = scaled_chunk
        
        uf.unityTime[0][:] += true_session_start  
       
    else:
        print('session start marker not recognised')
        print('unable to align timings accurately for now')        
        
    new_deltas = np.diff(uf.unityTime[0])
    for i in range(np.shape(uf.unityData[0])[0]):
        uf.unityData[0][i][1] = new_deltas[i]
    
    
    for col in range(np.shape(uf.unityTrialTime[0])[1]):
        
        arr = uf.unityTime[0][uf.unityTriggers[0][col][1]-1:uf.unityTriggers[0][col][2]]
        arr = arr - arr[0]
        a = np.empty((np.shape(uf.unityTrialTime[0])[0],1))
        a [:] = np.nan
        for i in range(np.shape(uf.unityTrialTime[0])[0]):
            uf.unityTrialTime[0][i,col] = a[i]
        for j in range(np.shape(arr)[0]):
            uf.unityTrialTime[0][j,col] = arr[j]      
    
    
    #uf_n = uf.get_filename()
    #el_n = el.get_filename()
    #hkl.dump(uf,uf_n,'w')
    #hkl.dump(el,el_n,'w')
    uf.save()
    el.save()
    
    print('finish aligning objects')
            


        
        
