#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "rplpl"   # job name

## /SBATCH -p general # partition (queue)
#SBATCH -o rplpl-slurm.%N.%j.out # STDOUT
#SBATCH -e rplpl-slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -u -c "import PyHippocampus as pyh; import DataProcessingTools as DPT; import time; print(time.localtime()); DPT.objects.processDirs(None, pyh.RPLParallel, saveLevel=1); print(time.localtime());"
