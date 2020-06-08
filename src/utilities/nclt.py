import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import struct
import scipy.interpolate

from tf import transformations as tf_tran

def convert(x_s, y_s, z_s):
    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def read_vel_sync_file(filename):
    f_bin = open(filename, "r")

    hits = []

    while True:

        x_str = f_bin.read(2)
        if x_str == '': # eof
            break

        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert(x, y, z)

        hits += [[x, y, z]]

    f_bin.close()

    hits = np.asarray(hits)

    return np.transpose(hits)

def ssc_to_homo(ssc):
    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation
    sr = np.sin(ssc[3])
    cr = np.cos(ssc[3])

    sp = np.sin(ssc[4])
    cp = np.cos(ssc[4])

    sh = np.sin(ssc[5])
    ch = np.cos(ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H

def get_gt_pose(gt, timestamp):
    func = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0)
    return func(timestamp)

def read_ground_truth(gt_dir):

    gt_np = np.loadtxt(gt_dir, delimiter=",")

    gt = dict()

    for i in range(gt_np.shape[0]):
        keyi = str(int(gt_np[i, 0]/1e4))
        gti = gt_np[i, 1:]

        gt[keyi] = gti

    print "the ground truth data is loaded"

    return gt

# Read ground truth data (first column is timestamp)
print 'Reading ground truth'
gt_nov = pd.read_csv('/media/kevin/ba404f82-4063-4ba5-8c0b-2c695fa22122/michigan/raw_data/2012-11-17/ground_truth.csv').values
gt_dec = pd.read_csv('/media/kevin/ba404f82-4063-4ba5-8c0b-2c695fa22122/michigan/raw_data/2012-12-01/ground_truth.csv').values

# Read velodyne data
print 'Reading velodyne sync data'

# Near beginning of logs
#t_nov = 1353174154883250
#t_dec = 1354396870271215

# A few hundred feet in
#t_nov = 1353174254271110
#t_dec = 1354396969260777

# Near Hayward Street and Draper Drive
t_nov = 1353174574831812
t_dec = 1354397273428767
vel_nov = read_vel_sync_file('/media/kevin/ba404f82-4063-4ba5-8c0b-2c695fa22122/michigan/raw_data/2012-11-17/velodyne_sync/%d.bin' % t_nov)
vel_dec = read_vel_sync_file('/media/kevin/ba404f82-4063-4ba5-8c0b-2c695fa22122/michigan/raw_data/2012-12-01/velodyne_sync/%d.bin' % t_dec)

# Get pose at each time according to ground truth
x_nov = get_gt_pose(gt_nov, t_nov)
x_dec = get_gt_pose(gt_dec, t_dec)

# Get transformations
H_global_nov1 = ssc_to_homo(x_nov)
H_global_dec1 = ssc_to_homo(x_dec)

print H_global_nov1

# Get velodyne points in homogenous coordinates
homo_nov = np.ones((1, vel_nov.shape[1]))
homo_dec = np.ones((1, vel_dec.shape[1]))
v1 = np.append(vel_nov, homo_nov, axis=0)
v2 = np.append(vel_dec, homo_dec, axis=0)


gt_nov2 = read_ground_truth('/media/kevin/ba404f82-4063-4ba5-8c0b-2c695fa22122/michigan/raw_data/2012-11-17/ground_truth.csv')
gt_dec2 = read_ground_truth('/media/kevin/ba404f82-4063-4ba5-8c0b-2c695fa22122/michigan/raw_data/2012-12-01/ground_truth.csv')

key = str(int(t_nov / 1e4))
[px_gt, py_gt, pz_gt, rx_gt, ry_gt, rz_gt] = gt_nov2[key]

T_t = tf_tran.translation_matrix((px_gt, py_gt, pz_gt))
T_r = tf_tran.euler_matrix(rx_gt, ry_gt, rz_gt)
H_global_nov = np.dot(T_t, T_r)

key = str(int(t_dec / 1e4))
[px_gt, py_gt, pz_gt, rx_gt, ry_gt, rz_gt] = gt_dec2[key]

T_t = tf_tran.translation_matrix((px_gt, py_gt, pz_gt))
T_r = tf_tran.euler_matrix(rx_gt, ry_gt, rz_gt)
H_global_dec = np.dot(T_t, T_r)

# Transform velodyne into global frame
v1_global = np.dot(H_global_nov, v1)
v2_global = np.dot(H_global_dec, v2)

print H_global_nov

plt.figure(1)
plt.scatter(gt_nov[:, 2], gt_nov[:, 1], s=1, edgecolors='r', marker='.')
plt.scatter(gt_dec[:, 2], gt_dec[:, 1], s=1, edgecolors='b', marker='.')
plt.scatter(x_nov[1], x_nov[0], s=200, linewidth=4, c='r', marker='x')
plt.scatter(x_dec[1], x_dec[0], s=200, linewidth=4, c='b', marker='x')
plt.grid()
plt.axis('equal')
plt.title('Ground Truth Trajectories')
plt.xlabel('East (m)')
plt.ylabel('North (m)')

plt.figure(2)
plt.scatter(vel_nov[1, :], vel_nov[0, :], c=vel_nov[2, :], s=1, marker='.')
plt.title('Velodyne Scan November')
plt.grid()
plt.axis('equal')

plt.figure(3)
plt.scatter(vel_dec[1, :], vel_dec[0, :], c=vel_dec[2, :], s=1, marker='.')
plt.title('Velodyne Scan December')
plt.grid()
plt.axis('equal')

plt.figure(4)
plt.scatter(v1_global[1, :], v1_global[0, :], edgecolors='r', s=1, marker='.')
plt.scatter(v2_global[1, :], v2_global[0, :], edgecolors='b', s=1, marker='.')
plt.scatter(gt_nov[:, 2], gt_nov[:, 1], s=1, edgecolors='k', marker='.')
plt.scatter(gt_dec[:, 2], gt_dec[:, 1], s=1, edgecolors='k', marker='.')
plt.scatter(x_nov[1], x_nov[0], s=200, linewidth=4, c='r', marker='x')
plt.scatter(x_dec[1], x_dec[0], s=200, linewidth=4, c='b', marker='x')
plt.title('Velodyne Scans')
plt.xlabel('East (m)')
plt.ylabel('North (m)')
plt.grid()
plt.axis('equal')

plt.show()
