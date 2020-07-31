#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "eyelink"   # job name
#SBATCH --mail-user=shihcheng@nus.edu.sg   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# this script can be used like this:
# [picasso]$ cwd=`pwd`; for i in `find . -name "session01"`; 
#     do echo $i; cd $i; sbatch /data/src_shihcheng/PyHippocampus/eyelink-slurm.sh; 
#     cd $cwd; done
python -c "import PyHippocampus as pyh; pyh.Eyelink(saveLevel = 1)"
