#!/bin/bash

# expects cluster config to reside in ~/.parallelcluster/config
# to be run outside the master node, before shutting down the cluster
# ~/.parallelcluster/config will be automatically repointed to the most updated snapshot

# pass unique description to identify all your snapshots as first argument
# pass the total number of snapshots (backups and current) to keep
# cluster identifier as optional 3rd argument

if [ "$#" -lt 2 ]
then
	echo "insufficient arguments - description and number of snapshots to keep needed"
	exit 1
fi

vol=$(aws ec2 describe-volumes --filter Name=attachment.device,Values=/dev/sdb Name=tag:aws:cloudformation:stack-name,Values=*$3* --query "Volumes[].VolumeId" --output text)

temp=($(aws ec2 create-snapshot --volume-id $vol --description "$1" --output text))
snap_in_prog=${temp[3]}
echo "new snapshot id: $snap_in_prog"

if [[ -z "${snap_in_prog// }" ]]; then
	echo "ERROR: no snapshot found, invalid volume?"
	exit 1
fi

echo "saving snapshot"
a=2
until [ $a -lt 1 ]
do
	status=$(aws ec2 describe-snapshots --filters Name=snapshot-id,Values=$snap_in_prog --query "Snapshots[].State" --output text)
	if [ "$status" == "completed" ]
	then
		break
	fi
	if [ "$status" == "error" ]
	then
		echo "ERROR: snapshot failed to save, exiting"
		exit 1
	fi	
done
echo "snapshot saved"

keep=$2
all_snaps=($(aws ec2 describe-snapshots --owner self --filters Name=description,Values=$1 --query 'Snapshots[].[StartTime,SnapshotId]' --output text | sort -n | cut -f 2))
total_count=${#all_snaps[@]}
most_recent=${all_snaps[$total_count-1]}

if [ "$total_count" -ge "$(($keep+1))" ]
then
	remove_snaps=(${all_snaps[@]:0:(($total_count-$keep))})
	for snap in ${remove_snaps[@]}
	do
		aws ec2 delete-snapshot --snapshot-id $snap
	done
fi

echo "keeping max $keep snapshots"
aws ec2 describe-snapshots --owner self --filters Name=description,Values=$1 --query 'Snapshots[].[StartTime,SnapshotId]' --output text | sort -n

sub_n=$(grep -n snap- ~/.parallelcluster/config | cut -d : -f 1)
head -$(($sub_n-1)) ~/.parallelcluster/config > ~/.parallelcluster/config.temp
echo "ebs_snapshot_id = $most_recent" >> ~/.parallelcluster/config.temp
tail -n +$(($sub_n+1)) ~/.parallelcluster/config >> ~/.parallelcluster/config.temp

mv ~/.parallelcluster/config.temp ~/.parallelcluster/config

echo "config file updated"

