#!/Volumes/User/shihcheng/anaconda3/bin/python

from filelock import FileLock
import hickle
import sys

# get lock
file_path = "envlist.hkl"
lock_path = "envlist.khl.lock"
time_out_secs = 60

# program modes
READ_MDODE = 0
WRITE_MODE = 1
RESET_MODE = 2

# get number of arguments
nargs = len(sys.argv)
if nargs > 3:
    print('Usage: envlist; envlist env; envlist reset nenvs')
    exit
elif nargs > 2:
    pmode = RESET_MODE
    nenvs = sys.argv[2]
elif nargs > 1:
    pmode = WRITE_MODE
    env = sys.argv[1]
else:
    pmode = READ_MODE

lock = FileLock(lock_path, timeout=time_out_secs)

with lock:
    if pmode == RESET_MODE:
        clist1 = [*range(0,int(nenvs),1)]
        clist = ['env' + str(s) for s in clist1]
    else:
        # load hickle file
        clist = hickle.load(file_path)

        if pmode == WRITE_MODE:
            # append item to end of list
            clist.append(env)
        else:    
            # pop first item off list
            env = clist.pop(0)
            # return env name
            print(env)

    # save hickle file
    hickle.dump(clist, file_path, mode="w")
