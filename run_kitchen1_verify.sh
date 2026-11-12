#!/bin/bash
set -e
export ROS_DOMAIN_ID=$(( RANDOM % 233 ))
echo "Using ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
export TRAIN_EXTRA_ARGS="--iterations 100000 --streaming_multires_it_v_floor 0 --streaming_multires_map_fps_floor 0"
human_launch_scripts/05_run_offline_rosbag.sh ~/Downloads/kitchen1 --idle-timeout 60 --model-path output/wave2_verify_kitchen1
