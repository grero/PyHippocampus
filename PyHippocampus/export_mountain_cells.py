import os
from mountainlab_pytools import mdaio
import scipy.io
import csv
from .spiketrain import Spiketrain

def export_mountain_cells():
    
    print('expecting call from channel level')
    init_path = os.getcwd()
    current_path = os.getcwd()
    path_split = str.split(current_path,'/')
    if path_split[-2] == 'mountains':
        channel_path = path_split[-1]
        os.chdir('output')
    else:
        os.chdir('../../../mountains/')
        os.chdir(path_split[-1])
        channel_path = path_split[-1]
        os.chdir('output')

    original = mdaio.readmda('firings.mda')
    os.chdir('..')
    start_time = {}
    with open('start_indices.csv') as csvfile:
        csvreader = csv.reader(csvfile,delimiter = ',')
        linecount = 0
        for line in csvreader:
            if linecount == 0:
                temp = []
                for term in line[1:]:
                    temp.append(int(term.strip(" '][")))
            else:
                temp = []
                for term in line[1:]:
                    temp.append(term.strip(" ']["))

            start_time[line[0]] = temp        
            linecount += 1
    
    current_path = os.getcwd()
    ch = current_path.partition('channel')[1] + current_path.partition('channel')[2]
    total_ind = len(original[1])
    marker = 0
    count = 0
    
    for i in range(total_ind):
        if count == len(start_time.get('2'))-1:
            exporting_arr = [original[1][marker:total_ind],original[2][marker:total_ind]]
            location = os.getcwd()
            temp = location.split('/')
            upper_folder1 = location[0:len(location)-len(temp[len(temp)-1])-1]
            os.chdir(upper_folder1)
            temp1 = upper_folder1.split('/')
            upper_folder2 = upper_folder1[0:len(upper_folder1)-len(temp1[len(temp1)-1])-1]
            os.chdir(upper_folder2)
            session_path = upper_folder2 + '/' + start_time.get('2')[count]
            os.chdir(session_path)
            full_list = []
            for name in os.listdir("."):
                if os.path.isdir(name):
                    full_list.append(name)
            for folder in full_list:
                array_path = session_path + '/' + folder
                os.chdir(array_path)
                if os.path.isdir(channel_path):
                    os.chdir(array_path)
                    split_into_cells_intra_session(ch, exporting_arr, start_time.get('1')[count]);
                    os.chdir(location)
            break
        if original[1][i] >= start_time.get('1')[count+1]:          
            exporting_arr = [original[1][marker:i-1],original[2][marker:i-1]]
            location = os.getcwd()
            temp = location.split('/')
            upper_folder1 = location[0:len(location)-len(temp[len(temp)-1])-1]
            os.chdir(upper_folder1)
            temp1 = upper_folder1.split('/')
            upper_folder2 = upper_folder1[0:len(upper_folder1)-len(temp1[len(temp1)-1])-1]
            os.chdir(upper_folder2)
            session_path = upper_folder2 + '/' + start_time.get('2')[count]
            os.chdir(session_path)
            full_list = []
            for name in os.listdir("."):
                if os.path.isdir(name):
                    full_list.append(name)
            for folder in full_list:
                array_path = session_path + '/' + folder
                os.chdir(array_path)
                if os.path.isdir(channel_path):
                    os.chdir(array_path)
                    split_into_cells_intra_session(ch, exporting_arr, start_time.get('1')[count])
            marker = i
            count += 1
            os.chdir(location)
            
    os.chdir(init_path)
                    
                
            
def split_into_cells_intra_session(channel, two_layer_chunk, start_ind):
    current = os.getcwd()
    channel_path = current + '/' + channel
    os.chdir(channel_path)
    time_chunk = two_layer_chunk[0]
    time_list = []
    for time in time_chunk:
        time = time - start_ind + 1
        time = time / 30000
        time = time * 1000
        time_list.append(time)
    
    unique_cells = []
    unique_cell = set(two_layer_chunk[1])
    for cell in unique_cell:
        unique_cells.append(int(cell))
    assignment_layer = two_layer_chunk[1]
    os.system('rm -r cell0*')
    
    for i in range(0,len(unique_cells)):
        if i+1 < 10:
            cell_name = 'cell0' + str(i+1)
        else:
            cell_name = 'cell' + str(i+1)
        cell_path = channel_path + '/' + cell_name
        os.mkdir(cell_path)
        os.chdir(cell_path)
        times = []
        for j in range(len(assignment_layer)):
            if assignment_layer[j] == unique_cells[i]:
                times.append(time_list[j])
        try:
            print(os.getcwd())
            print('spikecount')
            print(len(times))
            with open('spiketrain.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(times)
            
            Spiketrain(saveLevel=1)

        except IOError:
            print("I/O error")         
                
        
    

#print(export_mountain_cells()) 
