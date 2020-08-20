#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=2
#SBATCH -J "example-job"   # job name
#SBATCH --mail-user=<your-email>@nus.edu.sg   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

python -c "import PyHippocampus as pyh; import DataProcessingTools as DPT; import os; import time; time.localtime(); DPT.objects.processDirs(None, pyh.RPLParallel, saveLevel=1); DPT.objects.processDirs(None, pyh.RPLSplit, saveLevel=1); DPT.objects.processDirs(None, pyh.RPLLFP, saveLevel=1); DPT.objects.processDirs(None, pyh.RPLHighPass, saveLevel=1); DPT.objects.processDirs(None, pyh.Unity, saveLevel=1); pyh.EDFSplit(saveLevel=1); os.chdir('session01'); pyh.aligning_objects(); pyh.raycast(1); time.localtime();"
