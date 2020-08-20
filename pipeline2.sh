#!/bin/bash

# first job - no dependencies, called from the day directory
# creates RPLParallel object
jid1=$(sbatch /data/src/PyHippocampus/rplparallel-slurm.sh)

# second job depends on the first job
# creates Unity and Eyelink objects, runs aligning_objects, and then raycasting
jid2=$(sbatch --dependency=afterok:${jid1##* } /data/src/PyHippocampus/unity-slurm.sh)

# third job - no dependencies, called from the day directory
# splits ns5 file in sessioneye 
jid3=$(sbatch /data/src/PyHippocampus/rse-slurm.sh)

# fourth set of jobs - depends on third job, called from the day directory
jid4=$(sbatch --dependency=afterok:${jid3##* } /data/src/PyHippocampus/rs1-slurm.sh)
jid5=$(sbatch --dependency=afterok:${jid3##* } /data/src/PyHippocampus/rs2-slurm.sh)
jid6=$(sbatch --dependency=afterok:${jid3##* } /data/src/PyHippocampus/rs3-slurm.sh)
jid7=$(sbatch --dependency=afterok:${jid3##* } /data/src/PyHippocampus/rs4-slurm.sh)

# put dependency for any job that will spawn more jobs here
sbatch --dependency=afterok:${jid4##* }:${jid5##* }:${jid6##* }:${jid7##* } ~/consol_jobs.sh
