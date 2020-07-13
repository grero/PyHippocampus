# function to be called from session level
# $ python -c "from PyHippocampus import raycast; raycast.raycast(100)"

import PyHippocampus as pyh
import numpy as np
import hdf5storage as h5s
import os

def input_conversion(eyemat=0, unitymat=0):
    
    if unitymat == 0:
        # unity file conversion
        data = {}
        u = pyh.Unity()
        uf = {'data': data}
        data['unityTriggers'] = np.double(u.unityTriggers[0])
        data['unityData'] = np.double(u.unityData[0])
        h5s.savemat('unityfile.mat',{'uf':uf}, format='7.3')
    
    if eyemat == 0:
        # eyelink file conversion
        data = {}
        el = {'data': data}
        # to be done when Eyelink fixed

def raycast(radius=20, eyemat=0, unitymat=0):

    # create supporting file (list of session(s))
    session_path = os.getcwd()
    sfile = open("slist.txt","w")
    sfile.write(session_path)
    sfile.write("\n")
    sfile.close()

    # creating data files
    input_conversion(eyemat, unitymat)

    os.system("/data/RCP/VirtualMaze.x86_64 -screen-height 1080 -screen-width 1920 -batchmode -sessionlist slist.txt -density 220 -radius " + str(radius) + " -logfile logs.txt")


