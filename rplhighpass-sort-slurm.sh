#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=1M   # memory per CPU core
#SBATCH -J "example-job"   # job name
#SBATCH --mail-user=<your-email>@nus.edu.sg   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


## /SBATCH -p general # partition (queue)
## /SBATCH -o rplhp-sort-slurm.%N.%j.out # STDOUT
## /SBATCH -e rplhp-sort-slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

python -c "import PyHippocampus as ph; ph.RPLHighPass(saveLevel = 1); ph.mountain_batch(); ph.export_mountain_cells()"