import matplotlib.pyplot as plt
import numpy as np
import pcl
from show import *
from tool import *
import cv2

points = np.asarray(pcl.load('/home/kevin/pointclouds/1529946803.972923040.pcd'))
lidar = points

HRES = 0.16        # horizontal resolution (assuming 20Hz setting)
VRES = 0.3        # vertical res
VFOV = (-24.9, 2.0) # (-10.67, 30.67) # Field of view (-ve, +ve) along vertical axis
Y_FUDGE = 5        # y fudge factor for velodyne HDL 64E

lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, val="depth",
                       saveto="pic/lidar_depth.png", y_fudge=Y_FUDGE)

lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV, val="height",
                       saveto="pic/lidar_height.png", y_fudge=Y_FUDGE)

lidar_to_2d_front_view(lidar, v_res=VRES, h_res=HRES, v_fov=VFOV,
                       val="reflectance", saveto="pic/lidar_reflectance.png",
                       y_fudge=Y_FUDGE)

#birds_eye_point_cloud(lidar, side_range=(-50, 50), fwd_range=(-50, 50), res=0.1, saveto="pic/lidar_pil_01.png")

im = point_cloud_to_panorama(points,
                             v_res=VRES,
                             h_res=HRES,
                             v_fov=VFOV,
                             y_fudge=Y_FUDGE,
                             d_range=(0, 50))

cv2.imwrite('front_view.png', im)
#plt.show()

