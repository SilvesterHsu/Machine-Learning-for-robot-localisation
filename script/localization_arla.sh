roslaunch graph_map visualize_graph_fuser.launch localization:=true michigan:=true &
rosrun self_localization localization_offline   --initial_noise 1 1 0.00 6.14 6.15 0  --score-cell-weight 0.0 --ms-delay 0  --disable-unwarp --n-particles 4000 --forceSIR  --map-switching-method closest_observation --min-range 2 --keyframe-min-distance 0.1 --keyframe-min-rot-deg 100 --key-frame-update --tf-base-link /state_base_link --tf-gt-link /state_base_link --tf_world_frame /world --localisation-algorithm-name submap_mcl --skip-frame 1 --base-name mcl-ndt --bag-file-path $BAG_LOCATION/arla_bags/localisation/filtered.bag  --map-file-path /home/dlao/.ros/maps/offarla-2012_gt=0_submap=0_sizexy=300_Z=15_intrchR=8_compR=0_res=0.7_maxSensd=130_keyF=1_d=0.2_deg=100_alpha=0_dl=0_xyzir=0_mpsu=0_mnpfg=3kmnp1.map  --data-set arla-2012-noisy --z-filter-height -1000000  --velodyne-config-file "$(rospack find graph_map)/config/velo32.yaml" --lidar-topic /velodyne_packets resolution-local-factor 1.0 --visualize  --save-results  --visualize-map
#--random-initial-pose
#mcl_ndt
#--step-control
#--step-control
#--tf-base-link /robot_odom_link
#--gt-localization
