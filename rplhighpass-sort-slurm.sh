#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "rplhps"   # job name
#SBATCH -L sortinglicense:1 # licenses

## /SBATCH -p general # partition (queue)
#SBATCH -o rplhps-slurm.%N.%j.out # STDOUT
#SBATCH -e rplhps-slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

python -u -c "import PyHippocampus as pyh; pyh.RPLHighPass(saveLevel = 1); from PyHippocampus import mountain_batch; mountain_batch.mountain_batch(); from PyHippocampus import export_mountain_cells; export_mountain_cells.export_mountain_cells();"
