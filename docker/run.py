import PyHippocampus as ph 
from PyHippocampus import mountain_batch, export_mountain_cells
import sys 
import os 

print(sys.argv)
path = sys.argv[2]
os.chdir('/mnt/' + path)
print(os.getcwd())
rh = ph.RPLHighPass(saveLevel = 1) 
mountain_batch.mountain_batch()
export_mountain_cells.export_mountain_cells()


