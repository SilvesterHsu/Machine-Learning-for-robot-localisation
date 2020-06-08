killall lidar_odom_queue
killall self_localization
killall nn_localization_Server
killall nn_localization_client

echo "Kill various ros nodes"


#rosrun self_localization lidar_odom_queue.py &

rosrun self_localization nn_localization_sever.py --model_dir /home/data/nn_models/gp 

#rosrun self_localization nn_localization_client.py &
