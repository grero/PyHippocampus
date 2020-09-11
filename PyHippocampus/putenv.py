#!/Volumes/User/shihcheng/anaconda3/bin/python

from filelock import FileLock
import hickle
import sys

# get lock
file_path = "envlist.hkl"
lock_path = "envlist.khl.lock"
time_out_secs = 60

lock = FileLock(lock_path, timeout=time_out_secs)

with lock:
    # load hickle file
    clist = hickle.load(file_path)

    # append item to end of list
    clist.append(sys.argv[1])

    # save hickle file
    hickle.dump(clist, file_path, mode="w")
