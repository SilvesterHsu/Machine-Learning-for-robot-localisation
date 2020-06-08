#Kevin setup, please comment this Daniels when you want to use this script
for date in 2012_03_31 2012_09_28 2012_02_04 2012_02_05 2012_02_12 2012_04_29 2012_05_11 2012_06_15 2012_08_04 2012_10_28 2012_11_16 2012_12_01

do

echo $date

BAG_DIR="/home/kevin/DATA/Michigan/gt/"
BAG_EXTENSION=".bag_edited.bag"

BAG_FOLDER="$BAG_DIR$date/"

rosrun self_localization nn_data_saver.py --saving_dir=/home/kevin/data/michigan_nn_data/$date &
PID_saver=$!

for i in $(seq 0 10)

do

#BAG_PATH="$BAG_FOLDER$i$BAG_EXTENSION"
#MAP_PATH="/home/kevin/data/map/#offmichigan_gt=1_submap=1_sizexy=200_Z=17_intrchR=60_compR=10_res=1_maxSensd=130_keyF=1_d=0.4_deg=150_alpha=0_dl=0_xyzir=0_mpsu=0_mnpfg=3kmnp1.map"
#HEATMAP_PATH="/home/kevin/data/map/heatmap.map"
#
#Daniels setup, please comment this Kevin when you want to use this script
#BAG_PATH=$BAG_LOCATION/michigan/2012_05_11/1.bag_edited.bag
#MAP_PATH=/mnt/disk0/maps/michigan/2012_02_02/offmichigan_gt=1_submap=1_sizexy=200_Z=17_intrchR=60_compR=10_res=1_maxSensd=130_keyF=1_d=0.4_deg=150_alpha=0_dl=0_xyzir=0_mpsu=0_mnpfg=3kmnp1.map
#HEATMAP_PATH=/mnt/disk0/maps/michigan/january/heatmap.hmp
#KEVINS setup, please uncomment this Daniel when you want to use this script
#MAP_PATH="/home/kevin/DATA/Michigan/map/2012_05_11/offmichigan_gt=1_submap=1_sizexy=200_Z=17_intrchR=60_compR=10_res=1_maxSensd=130_keyF=1_d=0.5_deg=150_alpha=0_dl=0_xyzir=0_mpsu=0_mnpfg=3kmnp1.map"
#BAG_PATH="/home/kevin/DATA/Michigan/gt/2012_03_31/2.bag_edited.bag"



#if pgrep -x "rviz" > /dev/null
#then
#    echo "Rviz already running"
#else
#    rosrun rviz rviz -d "$(rospack find self_localization)/rviz/vis.rviz"&
#
#fi
rosrun self_localization localization_offline  --disable-localization --convergance-rate 0.5 --bag-start-index 0  --visualize --save-heatmap --load-heatmap --heatmap-path $HEATMAP_PATH --initial_noise 2 2 2 0 0 0    --init-ex 3.1415 --score-cell-weight 0.1 --ms-delay 0  --disable-unwarp  --n-particles 1000 --forceSIR  --map-switching-method closest_observation --min-range 2 --keyframe-min-distance 0 --keyframe-min-rot-deg 100 --key-frame-update --tf-base-link /robot_odom_link --tf-gt-link /state_base_link --localisation-algorithm-name submap_mcl --skip-frame 1 --base-name mcl-ndt --bag-file-path $BAG_PATH --map-file-path $MAP_PATH  --data-set michigan-noisy --z-filter-height -1000000  --velodyne-config-file "$(rospack find graph_map)/config/velo32.yaml" --lidar-topic /velodyne_points resolution-local-factor 2.0   --save-results


done

kill $PID_saver

done
exit 0


 #rosrun self_localization localization_offline  --uniform-particle-initialization --convergance-rate 0.5 --bag-start-index 2800  --visualize --save-heatmap --load-heatmap --heatmap-path $HEATMAP_PATH --initial_noise 2 2 2 0 0 0    --init-ex 3.1415 --score-cell-weight 0.1 --ms-delay 0  --disable-unwarp  --n-particles 1000 --forceSIR  --map-switching-method closest_observation --min-range 2 --keyframe-min-distance 0 --keyframe-min-rot-deg 100 --key-frame-update --tf-base-link /robot_odom_link --tf-gt-link /state_base_link --localisation-algorithm-name submap_mcl --skip-frame 1 --base-name mcl-ndt --bag-file-path $BAG_PATH --map-file-path $MAP_PATH  --data-set michigan-noisy --z-filter-height -1000000  --velodyne-config-file "$(rospack find graph_map)/config/velo32.yaml" --lidar-topic /velodyne_points resolution-local-factor 2.0   --save-results
#--step-control
#gt-localization
#--visualize --visualize-map
#--initial_noise 5 5 2 10 10 10

#roslaunch graph_map visualize_graph_fuser.launch localization:=true michigan:=true &
#rosrun self_localization localization_offline --nn-estimates --bag-start-index 0 --random-initial-pose  --init-ex 3.1415   --initial_noise 5 5 0 0 0 6  --score-cell-weight 0.0 --step-control --ms-delay 0  --disable-unwarp  --n-particles 1000 --forceSIR  --map-switching-method closest_observation --min-range 2 --keyframe-min-distance 0 --keyframe-min-rot-deg 100 --key-frame-update --tf-base-link /robot_odom_link --tf-gt-link /state_base_link --localisation-algorithm-name submap_mcl --skip-frame 1 --base-name mcl-ndt --bag-file-path $BAG_LOCATION --map-file-path $MAP_LOCATION  --data-set michigan-noisy --z-filter-height -1000000  --velodyne-config-file "$(rospack find graph_map)/config/velo32.yaml" --lidar-topic /velodyne_points    resolution-local-factor 1.5 --visualize --visualize-map --save-results #
#/home/dlao/.ros/maps/offmichigan_gt=0_submap=1_sizexy=200_Z=17_intrchR=30_compR=10_res=1_maxSensd=130_keyF=1_d=0.1_deg=150_alpha=0_dl=0_xyzir=0_mpsu=0_mnpfg=3kmnp1.map
#/home/dlao/.ros/maps/offmichigan_gt=1_submap=1_sizexy=200_Z=17_intrchR=60_compR=10_res=1_maxSensd=130_keyF=1_d=0.1_deg=150_alpha=0_dl=0_xyzir=0_mpsu=0_mnpfg=3kmnp1.map
#rosrun self_localization localization_offline --disable-unwarp --init-ex 3.1415 --n-particles 300 --forceSIR  --map-switching-method closest_observation --min-range 3 --keyframe-min-distance 0.1 --keyframe-min-rot-deg 100 --key-frame-update --tf-base-link /state_base_link --tf-gt-link /state_base_link --localisation-algorithm-name mcl_ndt --skip-frame 1 --base-name mcl-ndt --bag-file-path $BAG_LOCATION/michigan/2012_05_11/mapping/1.bag_edited.bag --map-file-path $MAP_LOCATION/michigan/2012_05_11_map/michigan_short_2012_05_11  --data-set michigan --z-filter-height -1000000  --velodyne-config-file "$(rospack find graph_map)/config/velo32.yaml" --lidar-topic /velodyne_points    resolution-local-factor 1.0 --visualize --visualize-map --save-results #
#mcl_ndt
#--step-control
#--tf-base-link /robot_odom_link
#--gt-localization
#
#("load-heatmap", "Load previous heatmap")
#("save-heatmap", "Save current heatmap")
#("heatmap-path", po::value<std::string>(&heatmap_path)->default_value(std::string("")), "path to heatmap, this reduces loading times")
#
#
