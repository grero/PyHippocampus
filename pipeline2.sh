#!/bin/bash

# first job - no dependencies, called from the day directory
# creates RPLParallel object
jid1=$(sbatch /data/src/PyHippocampus/rplparallel-slurm.sh)

# second job depends on the first job
# creates Unity and Eyelink objects, runs aligning_objects, and then raycasting
sbatch --dependency=afterok:${jid1##* } /data/src/PyHippocampus/unity-slurm.sh

# third job - no dependencies, called from the day directory
# splits ns5 file in sessioneye 
jid2=$(sbatch /data/src/PyHippocampus/rse-slurm.sh)

# fourth set of jobs - depends on third job, called from the day directory
sbatch --dependency=afterok:${jid2##* } /data/src/PyHippocampus/rs1-slurm.sh
sbatch --dependency=afterok:${jid2##* } /data/src/PyHippocampus/rs2-slurm.sh
sbatch --dependency=afterok:${jid2##* } /data/src/PyHippocampus/rs3-slurm.sh
sbatch --dependency=afterok:${jid2##* } /data/src/PyHippocampus/rs4-slurm.sh
