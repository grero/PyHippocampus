#! /bin/bash

# first job - no dependencies, called from the day directory
jid1=$(sbatch rplparallel-slurm.sh)

# second set of jobs - no dependencies, called from the day directory
sbatch rs1-slurm.sh
sbatch rs2-slurm.sh
sbatch rs3-slurm.sh
sbatch rs4-slurm.sh

# third job depends on the first job
jid3=$(sbatch --dependency=afterok:$jid1 unity-slurm.sh)
