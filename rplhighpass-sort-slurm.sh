#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -J "rplhps"   # job name

## /SBATCH -p general # partition (queue)
#SBATCH -o rplhps-slurm.%N.%j.out # STDOUT
#SBATCH -e rplhps-slurm.%N.%j.err # STDERR

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

conda init bash 
source ~/.bashrc
conda activate env1
curr_env=$CONDA_DEFAULT_ENV
new_env=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 4)
conda create --name $new_env --clone $curr_env 
conda activate $new_env
python -u -c "import PyHippocampus as pyh; pyh.RPLHighPass(saveLevel = 1); from PyHippocampus import mountain_batch; mountain_batch.mountain_batch(); from PyHippocampus import export_mountain_cells; export_mountain_cells.export_mountain_cells();"
conda deactivate 
conda remove --name $env --all -y 


