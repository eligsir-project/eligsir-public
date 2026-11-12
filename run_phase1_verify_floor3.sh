#!/bin/bash
set -e
rm -rf output/phase1_verify_floor3
export ROS_DOMAIN_ID=$(( RANDOM % 233 ))
echo "Using ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
scripts/run_offline_rosbag.sh /data/bags/floor3 --idle-timeout 30 --model-path output/phase1_verify_floor3
