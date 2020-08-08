#!/bin/bash

# first job - no dependencies, called from the day directory
jid1=$(sbatch /data/src/PyHippocampus/rplparallel-slurm.sh)

# second set of jobs - no dependencies, called from the day directory
sbatch /data/src/PyHippocampus/rs1-slurm.sh
sbatch /data/src/PyHippocampus/rs2-slurm.sh
sbatch /data/src/PyHippocampus/rs3-slurm.sh
sbatch /data/src/PyHippocampus/rs4-slurm.sh

# third job depends on the first job
jid3=$(sbatch --dependency=afterok:$jid1 /data/src/PyHippocampus/unity-slurm.sh)
