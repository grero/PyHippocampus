# function to be called from session level
# $ python -c "from PyHippocampus import raycast; raycast.raycast(100)"

import PyHippocampus as pyh
import numpy as np
import hdf5storage as h5s
import os

def input_conversion():
    
    if True:
        # unity file conversion
        data = {}
        u = pyh.Unity()
        uf = {'data': data}
        data['unityTriggers'] = np.squeeze(np.double(u.unityTriggers[0]))
        data['unityData'] = np.squeeze(np.double(u.unityData[0]))
        h5s.savemat('unityfile.mat',{'uf':uf}, format='7.3')
    
    if True:
        # eyelink file conversion
        data = {}
        eyedata = pyh.Eyelink()
        el = {'data': data}
        data['trial_timestamps'] = np.array(eyedata.trial_timestamps*1000)
        data['trial_codes'] = np.uint32(np.array(eyedata.trial_codes))
        data['timestamps'] = np.transpose(np.uint32(np.array(eyedata.timestamps*1000)[np.newaxis]))
        data['eye_pos'] = np.single(np.array(eyedata.eye_pos))
        h5s.savemat('eyelink.mat',{'el':el}, format='7.3')

def raycast(radius=20, eyemat=0, unitymat=0):

    # create supporting file (list of session(s))
    session_path = os.getcwd()
    sfile = open("slist.txt","w")
    sfile.write(session_path)
    sfile.write("\n")
    sfile.close()

    # creating data files
    input_conversion()

    os.system("/data/RCP/VirtualMaze.x86_64 -screen-height 1080 -screen-width 1920 -batchmode -sessionlist slist.txt -density 220 -radius " + str(radius) + " -logfile logs.txt")

    os.system("rm eyelink.mat unityfile.mat")

