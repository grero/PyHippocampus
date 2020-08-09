#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "rs2"   # job name

## /SBATCH -p general # partition (queue)
## /SBATCH -o rs2-slurm.%N.%j.out # STDOUT
## /SBATCH -e rs2-slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
python -u -c "import PyHippocampus as pyh; import DataProcessingTools as DPT; import time; print(time.localtime()); DPT.objects.processDirs(None, pyh.RPLSplit, channel=[*range(33,65)], SkipHPC=False, HPCScriptsDir = '/data/src/PyHippocampus/', SkipSort=False); print(time.localtime());"

