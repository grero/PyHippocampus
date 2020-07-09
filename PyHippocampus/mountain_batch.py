import numpy as np
import os 
import scipy.io
import csv
from mountainlab_pytools import mdaio


def mountain_batch(target,Overwrite):
    c_p = os.getcwd()
    split = c_p.split('/')
    if split[-1][0:7] == 'channel':
        target = split[-1][8:]
        day_dir = split[1:len(split)-3]
        day_path = ''
        for folder in day_dir:
            day_path =  day_path + '/' + folder 
        os.chdir(day_path)
    my_lst = ['all','hippo']
    if target in my_lst or type(target) is str:
        path = os.getcwd()
        n_path = path + '/mountains'
        print(n_path)
        if Overwrite == 'No':
            try:
                os.mkdir(n_path)
            except OSError:
                print("Creation of directory 'mountain' failed, folder already exists")
            else:
                print("Successfully created the directory 'mountains'")
        elif Overwrite == 'Yes':
            print('Overwriting existing "mountains" folder')
            os.system('rm -r mountains')  
            os.mkdir(n_path)
            print("Successfully overwritten the directory 'mountains'")
        else:
            print('Unspecified whether overwrite folder or not, pass "Yes" or "No" in the argument  of the function')
            return
    else:
        print('error function, specify arguments - all, hippo, <numbers>')
        return
    
    full_cell = comb_channels_ms()
    for i in range(len(full_cell.keys())):
        channel_no = full_cell.get(i)[0][8:]
        
        if target == 'all':
            mountain_channel(full_cell, i)
        
        elif target == 'hippo':
            for j in range(2,full_cell.get(i)[1]+2):
                session_extract = full_cell.get(i)[j].partition('session')[2]
                print(session_extract)
                if session_extract[0] == '0':
                    mountain_channel(full_cell,i)
                    break
        else:
            splits = target.split(' ')
            for k in range(0,len(splits)):
                if splits[k] == channel_no:
                    mountain_channel(full_cell,i)
                    break                    
                    
    
def comb_channels_ms():
    origin = os.getcwd()
    full_list = []
    for name in os.listdir("."):
        if os.path.isdir(name):
            full_list.append(name)
    top_folders = []
    session_names = []
    for i in range(1,len(full_list)):
        top_folders.append(full_list[i])
    for folder in top_folders:
        if len(folder) < 8:
            continue
        elif folder[0:8] == 'session0' or folder[0:10] == 'sessioneye':
            session_names.append(folder)
    
    day_path = os.getcwd()
    channels_identified = dict()
    identified_count = 0
    
    for i in range(len(session_names)):
        session_path = day_path + '/' + session_names[i]
        os.chdir(session_path)
        sub_list = []
        for name in os.listdir("."):
            if os.path.isdir(name):
                sub_list.append(name)        
        sub_folders = []
        array_folders = []
        for j in range(len(sub_list)):
            sub_folders.append(sub_list[j])
        for folder in sub_folders:
            if folder[0:5] == 'array':
                array_folders.append(folder)

        for k in range(len(array_folders)):
            channel_path = day_path + '/' + session_names[i] + '/' +array_folders[k]
            os.chdir(channel_path)
            channel_list = []
            for name in os.listdir("."):
                if os.path.isdir(name):
                    channel_list.append(name)             
            channel_folders = []
            for l in range(len(channel_list)):
                channel_folders.append(channel_list[l])
            for folder in channel_folders:
                if folder[0:7] == 'channel':
                    found = 0
                    if identified_count == 0:
                            info = []
                            info.append(folder)
                            info.append(1)
                            info.append(channel_path)
                            channels_identified[identified_count] = info
                            identified_count += 1
                            found = 1
                        
                    else:
                        for m in range(identified_count):
                            if channels_identified.get(m):
                                if channels_identified.get(m)[0] == folder:
                                    temp = channels_identified.get(m)
                                    temp.append(channel_path)
                                    temp[1] += 1
                                    channels_identified[m] = temp
                                    found = 1
                    if found == 0:
                        info1 = []
                        info1.append(folder)
                        info1.append(1)
                        info1.append(channel_path) 
                        channels_identified[identified_count] = info1
                        identified_count += 1
            os.chdir(session_path)
        os.chdir(day_path) 
    os.chdir(origin)
    return channels_identified
            
                        
                                
def mountain_channel(full_cell, index):
    print('processing channel')
    origin = os.getcwd()
    mountains_path = origin + '/mountains'
    os.chdir(mountains_path)
    dir_name = full_cell.get(index)[0]
    dir_path = mountains_path + '/' + dir_name
    if not os.path.isdir(dir_name):
        os.mkdir(dir_path)
    else:
        print('overwriting existing folder in mountains')
        print(os.getcwd())
        os.system('rm -r ' + dir_name)
        os.mkdir(dir_path)
    os.chdir(dir_path)
    list1 = []
    list2 = []
    count = 0
    start_indices = {1 : list1, 2 : list2}
    current = 1  
    current_path = os.getcwd()
    for n in range(2,full_cell.get(index)[1]+2):
        os.chdir(full_cell.get(index)[n]+ '/' + full_cell.get(index)[0])
        #os.chdir(full_cell.get(index)[0])
        print('debugging clones!')
        #print(os.getcwd())
        temp = start_indices.get(1)
        temp.append(current)
        start_indices[1] = temp
        cut = full_cell.get(index)[n].split('/')
        temp1 = start_indices.get(2)
        temp1.append(cut[len(cut)-2])
        start_indices[2] = temp1
        if n == 2:
            data = scipy.io.loadmat('rplhighpass.mat')
            data1 = data['rh'][0,0][0][0][0][0]
            current = len(data1)+1
        else:
            temp = scipy.io.loadmat('rplhighpass.mat')
            temp = temp['rh'][0,0][0][0][0][0]
            data1 = np.concatenate((data1,temp))
            current = len(data1)+1
    
    data1 = np.atleast_2d(data1).T 
    #print(start_indices)
    #print(data1.shape)
    os.chdir(current_path)

    try:
        with open('start_indices.csv', 'w') as f:
            for key in start_indices.keys():
                f.write("%s,%s\n"%(key,start_indices.get(key)))
    except IOError:
        print("I/O error")        
          
    dataset_path = current_path + '/dataset'
    try:
        os.mkdir(dataset_path)
    except OSError:
        print("Creation of directory 'dataset' failed")
    else:
        print("Successfully created the directory 'dataset'")    
    
    os.chdir(dataset_path)
    mdaio.writemda(data1, 'raw_data.mda', dtype='float32')
    x = mdaio.readmda('raw_data.mda')
    #print(x.shape)
    os.system('cp /home/carlos/temp/dataset/geom.csv .')
    #os.system('cp /home/ec2-user/geom.csv .')
    os.chdir(current_path)
    #os.system('cp /home/ec2-user/sort.sh.txt .')
    os.system('cp /home/carlos/temp/sort.sh.txt .')
    #print(os.getcwd())
    os.system('sh sort.sh.txt') 
    
    print('finish for this channel')
    os.chdir(origin)
            
            

#print(mountain_batch('34',Overwrite='No'))
        

    
    
                                
                                
                            
