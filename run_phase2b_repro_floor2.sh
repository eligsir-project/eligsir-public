#!/bin/bash
set -e
export ROS_DOMAIN_ID=$(( RANDOM % 233 ))
echo "Using ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
export TRAIN_EXTRA_ARGS="--online_mode --streaming_max_iterations 4000 --energy_dead_mode energy_quantile --streaming_debt_replay --streaming_consolidation_sampler debt --sh_impl gsplat --streaming_anneal_horizon 15000 --cap_max 600000 --streaming_target_capacity 200000 --loop_closure_enabled --loop_closure_apply_correction --loop_force_field_interval 500 --loop_closure_debug --streaming_multires_it_v_floor 0 --streaming_multires_map_fps_floor 0"
scripts/run_offline_rosbag.sh /data/bags/floor2 --idle-timeout 30 --model-path output/phase2c_repro_floor2_v1
