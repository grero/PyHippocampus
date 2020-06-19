#!/bin/bash

# expects cluster config to reside in ~/.parallelcluster/config
# to be run outside the master node, before shutting down the cluster
# ~/.parallelcluster/config will be automatically repointed to the most updated snapshot

# pass unique description to identify all your snapshots as first argument
# pass the total number of snapshots (backups and current) to keep

if [ "$#" -ne 2 ]
then
	echo "insufficient arguments - description and number of snapshots to keep needed"
	exit 1
fi

vol=$(aws ec2 describe-volumes --filter Name=attachment.device,Values=/dev/sdb --query "Volumes[].VolumeId" --output text)

temp=($(aws ec2 create-snapshot --volume-id $vol --description "$1" --output text))
snap_in_prog=${temp[3]}
echo $snap_in_prog

echo "saving snapshot"
a=2
until [ $a -lt 1 ]
do
	status=$(aws ec2 describe-snapshots --filters Name=snapshot-id,Values=$snap_in_prog --query "Snapshots[].State")
	status1=$(echo $status | cut -d '"' -f 2)
	if [ "$status1" == "completed" ]
	then
		break
	fi	
done
echo "snapshot saved"

keep=$2
all_snaps=($(aws ec2 describe-snapshots --filters Name=description,Values=$1 --query 'Snapshots[].[StartTime,SnapshotId]' --output text | sort -n | sed 's/.*\t//'))
total_count=${#all_snaps[@]}
most_recent=${all_snaps[-1]}

if [ "$total_count" -ge "$(($keep+1))" ]
then
	remove_snaps=(${all_snaps[@]:0:(($total_count-$keep))})
	for snap in ${remove_snaps[@]}
	do
		aws ec2 delete-snapshot --snapshot-id $snap
	done
fi
echo "keeping max $keep snapshots"

sed -i "s/snap-.*/$most_recent/" ~/.parallelcluster/config

echo "config file updated"

