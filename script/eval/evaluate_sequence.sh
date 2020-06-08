#Daniels setup, please comment this Kevin when you want to use this script
SEQUENCE=$1 #2012_04_29
EVAL_PATH=/mnt/disk0/maps/michigan/eval/eval_$2/$SEQUENCE
mkdir -p $EVAL_PATH
BAG_FOLDER=$BAG_LOCATION/michigan/$SEQUENCE
MAP_PATH=$MAP_LOCATION/michigan/training/offmichigan_gt=1_submap=1_sizexy=200_Z=17_intrchR=60_compR=10_res=1_maxSensd=130_keyF=1_d=0.2_deg=150_alpha=0_dl=0_xyzir=0_mpsu=0_mnpfg=3kmnp1.map
HEATMAP_PATH=/mnt/disk0/maps/michigan/training/heatmap.hmp


dataset=michigan-noisy

if pgrep -x "rviz" > /dev/null
then
    echo "Rviz already running"
else
    rosrun rviz rviz -d "$(rospack find self_localization)/rviz/vis.rviz"&

fi
echo "evaluate: $SEQUENCE with args: ${3}"

BAG_FILES=( `ls $BAG_FOLDER` )
BAG_FILES_length=${#BAG_FILES[@]}
echo $my_array_length
for element in "${BAG_FILES[@]}"
do
   echo "${element}"
done
spacing=140 #for hybrid approach

for bag_file in "${BAG_FILES[@]}"
do
  BAG_PATH=$BAG_FOLDER/$bag_file
  #rosrun self_localization convergence_test --fixed-covariance --test-index-spacing $spacing --bag-file-path $BAG_PATH --sequence-name $SEQUENCE --bag-file $bag_file --nn-estimates --output-dir-name $EVAL_PATH  --convergence-th 0.75  --exit-at-convergence --bag-start-index 1 --save-heatmap --load-heatmap --heatmap-path $HEATMAP_PATH --initial_noise 2 2 2 0 0 0    --init-ex 3.1415 --score-cell-weight 0.1 --ms-delay 0  --disable-unwarp  --n-particles 500 --forceSIR  --map-switching-method closest_observation --min-range 2 --keyframe-min-distance 0 --keyframe-min-rot-deg 100 --key-frame-update --tf-base-link /robot_odom_link --tf-gt-link /state_base_link --localisation-algorithm-name submap_mcl --skip-frame 1 --base-name mcl-ndt --map-file-path $MAP_PATH  --data-set $dataset --z-filter-height -1000000  --velodyne-config-file "$(rospack find graph_map)/config/velo32.yaml" --lidar-topic /velodyne_points resolution-local-factor 1.1   --random-initial-pose --save-results --convergance-rate 0.6
  #rosrun self_localization convergence_test --test-index-spacing $spacing --bag-file-path $BAG_PATH --sequence-name $SEQUENCE --bag-file $bag_file --nn-estimates --output-dir-name $EVAL_PATH  --convergence-th 0.75  --exit-at-convergence --bag-start-index 1 --save-heatmap --load-heatmap --heatmap-path $HEATMAP_PATH --initial_noise 2 2 2 0 0 0    --init-ex 3.1415 --score-cell-weight 0.1 --ms-delay 0  --disable-unwarp  --n-particles 500 --forceSIR  --map-switching-method closest_observation --min-range 2 --keyframe-min-distance 0 --keyframe-min-rot-deg 100 --key-frame-update --tf-base-link /robot_odom_link --tf-gt-link /state_base_link --localisation-algorithm-name submap_mcl --skip-frame 1 --base-name mcl-ndt --map-file-path $MAP_PATH  --data-set $dataset --z-filter-height -1000000  --velodyne-config-file "$(rospack find graph_map)/config/velo32.yaml" --lidar-topic /velodyne_points resolution-local-factor 1.1   --random-initial-pose --save-results --convergance-rate 0.6
  rosrun self_localization convergence_test --test-index-spacing $spacing --bag-file-path $BAG_PATH --sequence-name $SEQUENCE --bag-file $bag_file --uniform-particle-initialization --output-dir-name $EVAL_PATH  --convergence-th 0.75  --exit-at-convergence --bag-start-index 1500 --save-heatmap --load-heatmap --heatmap-path $HEATMAP_PATH --initial_noise 2 2 2 0 0 0    --init-ex 3.1415 --score-cell-weight 0.1 --ms-delay 0  --disable-unwarp  --n-particles 500 --forceSIR  --map-switching-method closest_observation --min-range 2 --keyframe-min-distance 0 --keyframe-min-rot-deg 100 --key-frame-update --tf-base-link /robot_odom_link --tf-gt-link /state_base_link --localisation-algorithm-name submap_mcl --skip-frame 1 --base-name mcl-ndt --map-file-path $MAP_PATH  --data-set $dataset --z-filter-height -1000000  --velodyne-config-file "$(rospack find graph_map)/config/velo32.yaml" --lidar-topic /velodyne_points resolution-local-factor 2.0   --random-initial-pose --save-results --convergance-rate 0.6
done



