#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "unity"   # job name

## /SBATCH -p general # partition (queue)
#SBATCH -o unity-slurm.%N.%j.out # STDOUT
#SBATCH -e unity-slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -u -c "import PyHippocampus as pyh; import DataProcessingTools as DPT; import os; import time; print(time.localtime()); DPT.objects.processDirs(None, pyh.Unity, saveLevel=1); pyh.EDFSplit(); os.chdir('session01'); pyh.aligning_objects(); pyh.raycast(1); print(time.localtime());"
