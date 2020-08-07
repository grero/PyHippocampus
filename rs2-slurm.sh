#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=1M   # memory per CPU core
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
python -u -c "import PyHippocampus as pyh; import DataProcessingTools as DPT; import time; time.localtime(); DPT.objects.processDirs(None, pyh.RPLSplit, saveLevel=1, channel=[33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,51,53,54,55,56,57,58,59,60,61,62,63,64]); time.localtime();"

