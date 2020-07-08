import h5py
import time
import numpy as np
import numpy.matlib


def aligning_objects():
    threshold = 0.02

    uf = h5py.File('unity.hdf5', 'r+')
    #el = h5py.File('eyelink.hdf5', 'r+')
    el = h5py.File('eyelink.mat', 'r+')
    rp = h5py.File('rplparallel.hdf5', 'r')
    
    #print(uf.keys())
    #print(el.get('el').get('data').keys())
    #print(np.array(el.get('el').get('data').get('session_start')))
    #print(rp.keys())
    
    
    true_timestamps = np.array(rp.get('timeStamps')) * 1000
    a = np.shape(true_timestamps)
    true_timestamps = np.reshape(true_timestamps,a[0]*a[1],order='C')
    
    
    
    el_trial_timestamps_flat = np.array(el.get('el').get('data').get('trial_timestamps'))
    b = np.shape(el_trial_timestamps_flat)
    el_trial_timestamps_flat = np.reshape(el_trial_timestamps_flat,b[0]*b[1],order='F')
    
    
    uf_unityTriggers_flat = uf.get('unityTriggers')
    c = np.shape(uf_unityTriggers_flat)
    uf_unityTriggers_flat = np.reshape( uf_unityTriggers_flat,c[0]*c[1],order='C')
    
    dubious_counter = 0
    dubious_collector = []  
    
    saving_closest_fix_times = el.get('el').get('data').get('fix_times')[0:2,:]
    saving_closest_fix_times = np.reshape(saving_closest_fix_times,(np.shape(saving_closest_fix_times)[0]*np.shape(saving_closest_fix_times)[1]),order='F')

    
    
    ts = np.array(el.get('el').get('data').get('timestamps'))
    ts = np.transpose(ts)
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
        current_chunk = np.array(el.get('el').get('data').get('timestamps'))
        d = np.shape(current_chunk)
        current_chunk = np.reshape(current_chunk,d[0]*d[1],order='F')
        
        current_chunk = current_chunk[int(current_start)-1:int(current_end)]
        current_chunk = current_chunk.astype(float)
        current_diff = current_chunk[np.shape(current_chunk)[0]-1] - current_chunk[0]
        
        
        current_start_time = current_chunk[0]
        current_end_time = current_chunk[np.shape(current_chunk)[0]-1]
        current_chunk = (current_chunk - current_start_time)* true_diff/current_diff 
        current_chunk = current_chunk + current_start_time
        shifting_needed = current_chunk[np.shape(current_chunk)[0]-1] - current_end_time
        
        
        el.get('el').get('data').get('timestamps')[int(current_start)-1:int(current_end)] = np.uint32(current_chunk)
        el.get('el').get('data').get('timestamps')[0][int(current_end):np.shape(el.get('el').get('data').get('timestamps'))[1]] = el.get('el').get('data').get('timestamps')[0][int(current_end):np.shape(el.get('el').get('data').get('timestamps'))[1]]+shifting_needed       
        
        
        true_diff = true_diff/1000
        
        current_start = uf_unityTriggers_flat[j]+2
        current_end = uf_unityTriggers_flat[j+1]+2
        current_chunk = uf.get('unityTime')[current_start-1:current_end]
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
        
    
        uf.get('unityTime')[int(current_start)-1:int(current_end)] = current_chunk
        uf.get('unityTime')[int(current_end):np.shape(uf.get('unityTime'))[0]] = uf.get('unityTime')[int(current_end):np.shape(uf.get('unityTime'))[0]] + shifting_needed        
        
        
        if dubious == 1:
            prev_prev_start = uf_unityTriggers_flat[j-2]+1
            chunk_size = np.shape(uf.get('unityTime')[prev_prev_start:current_start])[0]
            uf.get('unityTime')[prev_prev_start:current_start] = numpy.matlib.repmat(uf.get('unityTime')[prev_prev_start],1,chunk_size)
            
            dubious_counter = dubious_counter + 1
            dubious_collector.append(j)
            print('but disparity between rpl and unity was quite large')
            print(discrep)
   
    
    print('dubious counter: ' + str(dubious_counter))
    print('dubious location(s): ' + str(dubious_collector))
    
    markers = np.array(rp.get('rawMarkers'))
    if markers[0] == 84: 
        true_session_start = np.array(rp.get('session_start_sec'))
        session_trial_duration = rp.get('timeStamps')[0][0] - true_session_start
        session_trial_duration = session_trial_duration * 1000
        
    
        finding_index  = 0
        print(np.shape(np.array(el.get('el').get('data').get('timestamps')))[1])
        for i in range(np.shape(np.array(el.get('el').get('data').get('timestamps')))[1]):
            if el.get('el').get('data').get('timestamps')[0][i] != el.get('el').get('data').get('session_start'):
                finding_index += 1
            else:
                break
        
        el_session_trial_chunk = el.get('el').get('data').get('timestamps')[0][finding_index:int(el.get('el').get('data').get('trial_timestamps')[0][0])]
        el_session_trial_chunk.astype(float)
        last_point = el_session_trial_chunk[np.shape(el_session_trial_chunk)[0]-1]
        first_point = el_session_trial_chunk[0]   
        scaled_chunk = ((el_session_trial_chunk - el_session_trial_chunk[0]) / float(last_point - first_point)) * session_trial_duration
        scaled_chunk = scaled_chunk + first_point
        shifting_needed = scaled_chunk[np.shape(scaled_chunk)[0]-1] - last_point

        el.get('el').get('data').get('timestamps')[0][el.get('el').get('data').get('timestamps')[0][0]:np.shape(el.get('el').get('data').get('timestamps'))[1]] = el.get('el').get('data').get('timestamps')[0][el.get('el').get('data').get('timestamps')[0][0]:np.shape(el.get('el').get('data').get('timestamps'))[1]] + shifting_needed
        el.get('el').get('data').get('timestamps')[0][finding_index:int(el.get('el').get('data').get('trial_timestamps')[0][0])] = scaled_chunk
        
        target = true_session_start * 1000
        full_shift = np.array(el.get('el').get('data').get('session_start')) - target
        el.get('el').get('data').get('timestamps')[:] = np.uint32(el.get('el').get('data').get('timestamps')[:]-full_shift)
       
        
        working_copy = el.get('el').get('data').get('fix_times')[0:2,:]
        working_copy = np.uint32(np.transpose(working_copy))
        
        TS = np.array(el.get('el').get('data').get('timestamps'))
        TS = np.transpose(TS)
        for row in range(np.shape(working_copy)[0]):
            for col in range(0,2):
                working_copy[row][col] = TS[int(saving_closest_fix_times[row][col])]          
        working_copy = np.transpose(working_copy)
        el.get('el').get('data').get('fix_times')[0:2,:] = working_copy

        session_trial_duration = rp.get('timeStamps')[0][0] - true_session_start
        uf_session_trial_chunk = uf.get('unityTime')[0:uf.get('unityTriggers')[0][0]+2]
        last_point = uf_session_trial_chunk[np.shape(uf_session_trial_chunk)[0]-1]
        scaled_chunk = (uf_session_trial_chunk/last_point) * session_trial_duration
        shifting_needed = scaled_chunk[np.shape(scaled_chunk)[0]-1] - last_point
        uf.get('unityTime')[uf.get('unityTriggers')[0][0]+1:np.shape(uf.get('unityTime'))[0]] += shifting_needed
        uf.get('unityTime')[0:uf.get('unityTriggers')[0][0]+2] = scaled_chunk
        
        uf.get('unityTime')[:] += true_session_start  
       
    else:
        print('session start marker not recognised')
        print('unable to align timings accurately for now')        
        
    
    new_deltas = np.diff(uf.get('unityTime'))
    for i in range(np.shape(uf.get('unityData'))[0]):
        uf.get('unityData')[i][1] = new_deltas[i]
    
    
    
    for col in range(np.shape(uf.get('unityTrialTime'))[1]):
        
        arr = uf.get('unityTime')[uf.get('unityTriggers')[col][1]-1:uf.get('unityTriggers')[col][2]]
        arr = arr - arr[0]
        a = np.empty((np.shape(uf.get('unityTrialTime'))[0],1))
        a [:] = np.nan
        for i in range(np.shape(uf.get('unityTrialTime'))[0]):
            uf.get('unityTrialTime')[i,col] = a[i]
        for j in range(np.shape(arr)[0]):
            uf.get('unityTrialTime')[j,col] = arr[j]      
    
    
    uf.close()
    el.close()
    rp.close()
            


#print(aligning_objects())

        
        
