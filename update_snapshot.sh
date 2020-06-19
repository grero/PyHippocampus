#!/bin/bash

vol=$(aws ec2 describe-volumes --filter Name=attachment.device,Values=/dev/sdb --query "Volumes[].VolumeId" --output text)

temp=($(aws ec2 create-snapshot --volume-id $vol --description "picasso-misc2" --output text))
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

keep=2
all_snaps=($(aws ec2 describe-snapshots --filters Name=description,Values=picasso-misc2 --query 'Snapshots[].[StartTime,SnapshotId]' --output text | sort -n | sed 's/.*\t//'))
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
echo "purged all but $keep snapshots"

sed -i "s/snap-.*/$most_recent/" ~/.parallelcluster/config

