
#SEQUENCE=2012_05_11
#BAG_FOLDER=$BAG_LOCATION/michigan/$SEQUENCE
EVAL_PATH=$1

#KEVINS setup, please uncomment this Daniel when you want to use this script
#MAP_PATH="/home/kevin/DATA/Michigan/map/2012_05_11/offmichigan_gt=1_submap=1_sizexy=200_Z=17_intrchR=60_compR=10_res=1_maxSensd=130_keyF=1_d=0.5_deg=150_alpha=0_dl=0_xyzir=0_mpsu=0_mnpfg=3kmnp1.map"
#BAG_PATH="/home/kevin/DATA/Michigan/gt/2012_03_31/2.bag_edited.bag"

#dataset=michigan-noisy
EVAL_OUTPUT_FOLDER=$EVAL_PATH/results
mkdir -p "$EVAL_OUTPUT_FOLDER"

output_file_merged=$EVAL_OUTPUT_FOLDER/eval_output_all.txt
#output_file_fixed=$EVAL_OUTPUT_FOLDER/eval_output_fixed_cov.txt
#output_file_dynamic=$EVAL_OUTPUT_FOLDER/eval_output_dynamic_cov.txt
#output_file_uniform=$EVAL_OUTPUT_FOLDER/eval_output_uniform.txt


i=1
folder_days=( `ls $EVAL_PATH |grep 2012` )



for day in "${folder_days[@]}"
do
fixed_cov_files=( `ls $EVAL_PATH/$day |grep fixed=1.*metadata` )
dynamic_cov_files=( `ls $EVAL_PATH/$day |grep fixed=0.*metadata` )
uniform_files=( `ls $EVAL_PATH/$day |grep uniform.*metadata` )

    for eval_file in "${fixed_cov_files[@]}"
    do
        if [[ "$i" == '1' ]];
        then
            output_str="$(head -n1 $EVAL_PATH/$day/$eval_file) fixed uniform day"
            echo $output_str >> $output_file_merged
        ((i++))
        fi
            output_str="$(tail -n1 $EVAL_PATH/$day/$eval_file) 1 0 $day"
            echo $output_str >> $output_file_merged

    done

    for eval_file in "${dynamic_cov_files[@]}"
    do
 	if [[ "$i" == '1' ]];
        then
            output_str="$(head -n1 $EVAL_PATH/$day/$eval_file) fixed uniform day"
            echo $output_str >> $output_file_merged
            ((i++))
	fi
        output_str="$(tail -n1 $EVAL_PATH/$day/$eval_file) 0 0 $day"
        echo $output_str >> $output_file_merged
    done

    for eval_file in "${uniform_files[@]}"
    do
 	if [[ "$i" == '1' ]];
        then
            output_str="$(head -n1 $EVAL_PATH/$day/$eval_file) fixed uniform day"
            echo $output_str >> $output_file_merged
	    ((i++))	
	fi
        output_str="$(tail -n1 $EVAL_PATH/$day/$eval_file) 0 1 $day"
        echo $output_str >> $output_file_merged
    done
done




#i=1
#for eval_file in "${dynamic_cov_files[@]}"
#do
#    if [[ "$i" == '1' ]];
#    then
#        output_str="$(head -n1 $EVAL_PATH/$eval_file) fixed uniform"
#        echo $output_str >> $output_file_dynamic
#    else
#        output_str="$(tail -n1 $EVAL_PATH/$eval_file) 0 0"
#        echo $output_str >> $output_file_dynamic
#    fi
#    ((i++))
#done
#i=1
#for eval_file in "${uniform_files[@]}"
#do
#    if [[ "$i" == '1' ]];
#    then
#        output_str="$(head -n1 $EVAL_PATH/$eval_file) fixed uniform"
#        echo $output_str >> $output_file_uniform
#    else
#        output_str="$(tail -n1 $EVAL_PATH/$eval_file) 0 1"
#        echo $output_str >> $output_file_uniform
#    fi
#((i++))
#done





#--uniform-particle-initialization
#--nn-estimates
# --uniform-particle-initialization
# --convergance-rate 0.5
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
