#!/usr/bin/env python

#import pcl
import numpy as np
import os
import random
from tqdm import tqdm
from PIL import Image
import torchlib.lidar_projection
import tf.transformations as tf_tran
import math
from tqdm import tqdm
import imutils


RESOLUTION = 0.4
PERCENTAGE = 1.0


def normalize(Targets):
    Targets = np.asarray(Targets)
    trans_data = Targets[:, :3]
    rot_data = Targets[:, -4:]
    # subtract mean
    norm_center = np.mean(trans_data, axis=0)
    centerized = trans_data - (np.tile(norm_center, (trans_data.shape[0], 1)))
    # divide standard deviation
    norm_std = np.std(centerized, axis=0)
    normalized = centerized / (np.tile(norm_std, (trans_data.shape[0], 1)))
    Target2 = np.concatenate([normalized, rot_data], axis=1)
    return Target2.tolist(), norm_center, norm_std

def de_normalize(Data, norm_center, norm_std):
    # multiply standard deviation
    dim = Data.shape[1]
    de_centered = Data * (np.tile(norm_std[0:dim], (Data.shape[0], 1)))
    # subtract mean
    de_normalized = de_centered + (np.tile(norm_center[0:dim], (Data.shape[0], 1)))

    return de_normalized


def get_per_class_accuracy(preds, labels, num_cat):

    preds = np.array(preds)
    labels = np.array(labels)

    acc = []

    for c in range(num_cat):
        if np.sum(labels==c) == 0:
            acc.append(1)
        else:
            acc.append(float(np.sum(preds[np.nonzero(labels==c)] == labels[np.nonzero(labels==c)])) / float(np.sum(labels==c)))

    return acc

def get_hash_key(pos_x, pos_y, pos_z):
    pos_x += 50
    pos_y += 50
    pos_z += 5

    gridCellsWide = 100 / RESOLUTION
    gridCellsHigh = 100 / RESOLUTION
    gridCellsDeep = 10 / RESOLUTION

    gridSize = RESOLUTION

    # encode
    x = pos_x / gridSize
    y = pos_y / gridSize
    z = pos_z / gridSize
    # key
    hash = x + (y * gridCellsWide) + (z * gridCellsWide * gridCellsHigh)

    return str(int(hash))


def cat2label(name):
    switcher = {
        'car': 1,
        'pedestrian': 2,
        'cyclist': 3,
        'dontcare': -1,
    }
    return switcher.get(name, 0)

def get_label(pt, annotation):

    offset = RESOLUTION / 1

    label = 0

    for a in annotation:
        bbox_min = a[0:3]
        bbox_max = a[3:6]

        if pt[0] < bbox_min[0] - offset or pt[1] < bbox_min[1] - offset or pt[2] < bbox_min[2] - offset or pt[0] > bbox_max[0] + offset or pt[1] > bbox_max[1] + offset or pt[2] > bbox_max[2] + offset:
            continue

        label = a[-1]
        break

    return label

def crop_box2(in_cloud, bbox_min, bbox_max):

    offset = RESOLUTION / 1

    out_cloud = []

    for pt in in_cloud:
        if pt[0] < bbox_min[0] - offset or pt[1] < bbox_min[1] - offset or pt[2] < bbox_min[2] - offset or pt[0] > bbox_max[0] + offset or pt[1] > bbox_max[1] + offset or pt[2] > bbox_max[2] + offset:
            out_cloud.append(pt)

    return out_cloud

def crop_box(in_cloud, bbox_min, bbox_max):

    offset = RESOLUTION / 1

    clipper = in_cloud.make_cropbox()
    clipper.set_MinMax(bbox_min[0]-offset, bbox_min[1]-offset, bbox_min[2]-offset, 0, bbox_max[0]+offset, bbox_max[1]+offset, bbox_max[2]+offset, 0)
    out_cloud = clipper.filter()

    return np.asarray(out_cloud)


def read_annotation_file(label_file):

    read_stream = open(label_file, 'r').read().splitlines()

    bboxes = []
    sizes = []

    for i in range(0, len(read_stream)):
        info = read_stream[i].split(" ")
        bbox = np.array([info[4], info[5], info[6], info[7], info[8], info[9], cat2label(info[0])], dtype=np.float)
        bboxes.append(bbox)
        tmp = bbox[0:3] - bbox[3:6]
        sizes.append(tmp.dot(tmp.T))

    idx = np.argsort(np.array(sizes))
    bboxes = [bboxes[i] for i in idx]

    return bboxes

def convert_bbox_to_point_wise(pcs, bbox_feats, annotations):

    frame_start_idx = 0
    day_end_inx = []
    pts_counter = 0
    recurrent_map = dict()

    for d, (pc, bbox_feat, annotation) in enumerate(zip(pcs, bbox_feats, annotations)):
        for i in tqdm(range(bbox_feat.shape[0])):

            line = bbox_feat[i, :]
            frame = line[0]

            ###bbox_min = line[1:4]
            ###bbox_max = line[4:7]
            bbox_min = [min(line[1], line[4]), min(line[2], line[5]), min(line[3], line[6])]
            bbox_max = [max(line[1], line[4]), max(line[2], line[5]), max(line[3], line[6])]

            prob = line[7:11]
            feat = line[11:]

            map_pts = crop_box(pc, bbox_min, bbox_max)

            if len(map_pts.shape) == 0:
                continue

            for j in range(map_pts.shape[0]):
                [pos_x, pos_y, pos_z] = map_pts[j, :]

                label = get_label([pos_x, pos_y, pos_z], annotation)
                key = get_hash_key(pos_x, pos_y, pos_z)

                if key not in recurrent_map:
                    recurrent_map[key] = []

                recurrent_map[key].append([frame_start_idx + frame] + [pos_x, pos_y, pos_z] + [d, i, label])
                pts_counter += 1

        frame_start_idx = bbox_feat[-1, 0]
        day_end_inx.append(frame_start_idx)

    print("number of observations (vexol-wise): " + str(pts_counter))
    print("number of recurrent cells:" + str(len(recurrent_map.keys())))

    print("done!")

    is_save = True
    if is_save:
        import csv

        w = csv.writer(open("recurrent_map.csv", "w"))
        for key, val in recurrent_map.items():
            w.writerow([key, val])

    return recurrent_map, day_end_inx

def load_all_data(map_dir, map_files, bbox_feat_dir, bbox_feat_files, annotation_dir, annotation_files):

    PC = []
    Bbox_Feat = []
    Annotation = []

    for mf, bf, af in tqdm(zip(map_files, bbox_feat_files, annotation_files)):
        # load maps
        # print "loading " + mf
        pc = pcl.load(os.path.join(map_dir, mf))
        # load detection bbox features
        # print "loading " + bf
        bbox_feat = np.loadtxt(os.path.join(bbox_feat_dir, bf))
        # load annotation info
        # print "loading " + af
        annotation = read_annotation_file(os.path.join(annotation_dir, af))

        PC.append(pc)
        Bbox_Feat.append(bbox_feat)
        Annotation.append(annotation)

    return PC, Bbox_Feat, Annotation


def image_argumentation(img, target, target_image_size=[200, 200], mode='train', rot_angle=None):
    RES = 100.0 / 400.0

    margin_row = img.shape[0] - target_image_size[0]
    margin_col = img.shape[1] - target_image_size[1]

    [px, py, pz, qx, qy, qz, qw] = target

    if mode == 'train':

        offset_row = int(max(0.0, min(1.0, np.random.normal(0.5, 0.1))) * margin_row)
        offset_col = int(max(0.0, min(1.0, np.random.normal(0.5, 0.1))) * margin_col)
        # offset_row = random.randint(0, margin_row)
        # offset_col = random.randint(0, margin_col)

        '''sensor offset'''
        # T_base2laser = tf_tran.euler_matrix(0.807*math.pi/180., 0.166*math.pi/180., -90.703*math.pi/ 180., 'rxyz')
        # T_base2laser[0:3, 3] = [0.002, -0.004, -0.957]
        # T_laser2base = np.linalg.inv(T_base2laser)

        '''argumentation'''
        offset_x = (offset_row - int(margin_row / 2)) * RES
        offset_y = -(offset_col - int(margin_col / 2)) * RES

        if rot_angle is None:
            # not rotate
            deltaT = tf_tran.identity_matrix()
        else:
            # rotate image 180 with noise
            angle = rot_angle

            deltaT = tf_tran.rotation_matrix(angle, (0,0,1))
            img = imutils.rotate(img[:, :, 0], angle/math.pi*180)
            img = np.array(img)[:, :, np.newaxis]

        deltaT[0:3, 3] = [offset_x, offset_y, 0.]

        # transform offset from /laser to /base
        # deltaT = np.matmul(deltaT, T_laser2base)

        '''gt'''
        T = tf_tran.quaternion_matrix([qx, qy, qz, qw])
        T[0:3, 3] = [px, py, pz]

        # apply on global pose
        T = np.matmul(deltaT, T)

        position = np.array(tf_tran.translation_from_matrix(T), dtype=np.single)
        quaternion = np.array(tf_tran.quaternion_from_matrix(T), dtype=np.single)

        target = np.concatenate((position, quaternion), axis=0)

    else:
        offset_row = int(margin_row / 2)
        offset_col = int(margin_col / 2)

    img = img[offset_row:offset_row + target_image_size[0], offset_col:offset_col + target_image_size[1], :]

    return img, target


class DataLoader:
    def __init__(self, batch_size=32,  dataset_dirs='', is_argumentation=False, target_image_size=None):

        self.batch_size = batch_size
        self.data_argumentation = is_argumentation
        self.target_image_size = target_image_size
        self.num_connected_frames = 3

        self.dataset_dirs = []
        self.file_indices = []

        for dataset_dir in dataset_dirs:
            text_file = open(os.path.join(dataset_dir, 'index.txt'), "r")

            file_index = text_file.readlines()
            self.file_indices += file_index
            self.dataset_dirs += [dataset_dir for i in range(len(file_index))]

        self.load_all_data(dataset_dirs)
        self.num_batches = int(len(self.idx) / batch_size * PERCENTAGE)


    def load_all_data(self, dataset_dirs):

        self.Images = []
        self.Targets = []

        self.idx = []
        tmp_idx = 0

        for di, (dataset_dir) in enumerate(dataset_dirs):
            text_file = open(os.path.join(dataset_dir, 'index.txt'), "r")

            file_index = text_file.readlines()
            file_index = file_index[1:-1:2]
            # file_index = random.sample(file_index, int(len(file_index) * 0.1))

            for fi, (index) in tqdm(enumerate(file_index)):
                img = np.array(Image.open(os.path.join(dataset_dir, 'images', index[:-1] + '.png')), dtype=np.uint8)
                img = img[:, :, np.newaxis]
                target = np.loadtxt(os.path.join(dataset_dir, 'poses', index[:-1] + '.txt'))
                [px, py, pz, ex, ey, ez] = target
                # [qw, qx, qy, qz] = tf_tran.quaternion_from_euler(ex, ey, ez, axes='sxyz')
                R = tf_tran.euler_matrix(ex, ey, ez, 'rxyz')
                [qx, qy, qz, qw] = tf_tran.quaternion_from_matrix(R)
                target = np.array([px, py, pz, qx, qy, qz, qw], dtype=np.float32)

                np.testing.assert_approx_equal(np.sum(target[-4:] ** 2), 1.0, significant=5)

                self.Images.append(img)
                self.Targets.append(target)

                if fi >= self.num_connected_frames:
                    self.idx.append(tmp_idx)

                tmp_idx += 1

        _, self.norm_mean, self.norm_std = normalize(self.Targets)

    def shuffle_data(self, mode='train'):

        self.idx = np.asarray(self.idx)
        #self.idx = np.arange(len(self.Targets)-1) + 1
        self.is_epoch_done = False
        self.pointer = 0

        if mode == 'train':
            np.random.shuffle(self.idx)


    def next_batch(self, b, mode='train'):

        start_idx = b * self.batch_size
        end_idx = (b + 1) * self.batch_size

        batch_idx = self.idx[start_idx:end_idx]

        batch_x = []
        batch_y = []

        for i in batch_idx:
            img = self.Images[i]
            target = self.Targets[i]
            #img_argumented, target_argumented = image_argumentation(img, target, target_image_size=[300, 300], mode=mode)
            img_argumented, target_argumented = image_argumentation(img, target, target_image_size=self.target_image_size, mode=mode)
            batch_x.append(img_argumented)
            batch_y.append(target_argumented)

        return batch_x, batch_y

    def next_pair_batch(self, b, mode='train'):

        start_idx = b * self.batch_size
        end_idx = (b + 1) * self.batch_size

        batch_idx = self.idx[start_idx:end_idx]

        batch_x0 = []
        batch_x1 = []
        batch_y0 = []
        batch_y1 = []

        for i in batch_idx:
            paired_frame_offset = random.randint(1, self.num_connected_frames)
            img0 = self.Images[i-paired_frame_offset]
            img1 = self.Images[i]
            target0 = self.Targets[i-paired_frame_offset]
            target1 = self.Targets[i]

            if random.random() > 0.5 and mode == 'train':
                rand_angle = math.pi + random.uniform(-1, 1) * math.pi / 180 * 15
                # rand_angle = math.pi
            else:
                rand_angle = None

            rand_angle = None

            #img_argumented0, target_argumented0 = image_argumentation(img0, target0, target_image_size=[300, 300], mode=mode, rot_angle=rand_angle)
            #img_argumented1, target_argumented1 = image_argumentation(img1, target1, target_image_size=[300, 300], mode=mode, rot_angle=rand_angle)

            img_argumented0, target_argumented0 = image_argumentation(img0, target0, target_image_size=self.target_image_size, mode=mode, rot_angle=rand_angle)
            img_argumented1, target_argumented1 = image_argumentation(img1, target1, target_image_size=self.target_image_size, mode=mode, rot_angle=rand_angle)

            np.testing.assert_approx_equal(np.sum(target_argumented0[-4:] ** 2), 1.0, significant=5)
            np.testing.assert_approx_equal(np.sum(target_argumented1[-4:] ** 2), 1.0, significant=5)

            # if np.isnan(target_argumented0).any() or np.isnan(target_argumented1).any():
            #     i -= 1
            #     continue

            batch_x0.append(img_argumented0)
            batch_x1.append(img_argumented1)
            batch_y0.append(target_argumented0)
            batch_y1.append(target_argumented1)

        return batch_x0, batch_x1, batch_y0, batch_y1

    def tick_pointer(self):
        self.pointer += 1

        if self.pointer == self.num_batches:
            self.pointer = 0
            self.is_epoch_done = True



class SequenceDataLoader:
    def __init__(self, batch_size=32, max_seq_length=10, dataset_dirs='', is_argumentation=False, target_image_size=None):

        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.data_argumentation = is_argumentation
        self.target_image_size = target_image_size

        self.load_all_data(dataset_dirs)
        self.num_batches = int(len(self.FileIndex) / batch_size * PERCENTAGE)


    def load_all_data(self, dataset_dirs):

        self.Images = []
        self.Targets = []
        self.DatasetIndex = []
        self.FileIndex = []
        self.SequenceEnd = []

        prev_end_inx = 0
        for di, (dataset_dir) in enumerate(dataset_dirs):
            text_file = open(os.path.join(dataset_dir, 'index.txt'), "r")

            file_index = text_file.readlines()

            for fi, (index) in tqdm(enumerate(file_index)):
                img = np.array(Image.open(os.path.join(dataset_dir, 'images', index[:-1] + '.png')), dtype=np.uint8)
                img = img[:, :, np.newaxis]
                target = np.loadtxt(os.path.join(dataset_dir, 'poses', index[:-1] + '.txt'))
                [px, py, pz, ex, ey, ez] = target
                [qx, qy, qz, qw] = tf_tran.quaternion_from_euler(ex, ey, ez)
                target = [px, py, pz, qx, qy, qz, qw]

                self.Images.append(img)
                self.Targets.append(target)
                self.DatasetIndex.append(di)
                self.FileIndex.append(prev_end_inx + fi)
                self.SequenceEnd.append(prev_end_inx + min(len(file_index), fi+random.randint(10, 100)))

            prev_end_inx += len(file_index)

        _, self.norm_mean, self.norm_std = normalize(self.Targets)




    def shuffle_data(self, mode='train'):

        assert len(self.FileIndex) == len(self.SequenceEnd) == len(self.DatasetIndex) == len(self.Images) == len(self.Targets), "Indices are not with the same length!"

        self.idx = np.arange(len(self.FileIndex))
        self.is_epoch_done = False
        self.pointer = 0
        self.batch_seq_start_idx = [None for _ in range(self.batch_size)]
        self.batch_seq_end_idx = [None for _ in range(self.batch_size)]

        if mode == 'train':
            np.random.shuffle(self.idx)

    def tick_pointer(self):
        self.pointer += 1
        #print(str(self.pointer) + " of " + str(int(len(self.FileIndex)*PERCENTAGE)))

        if self.pointer == int(len(self.FileIndex)*PERCENTAGE):
            self.pointer = 0
            self.is_epoch_done = True


    def next_rnn_batch(self):

        batch_x = []
        batch_y = []
        batch_seq_len = []
        is_feed_state = []

        # initialize new sequence for batch_data
        for i in range(self.batch_size):
            if self.batch_seq_start_idx[i] is None:
                self.batch_seq_start_idx[i] = self.FileIndex[self.idx[self.pointer]]
                self.batch_seq_end_idx[i] = self.SequenceEnd[self.idx[self.pointer]]
                self.tick_pointer()

            #seq_x = np.zeros([self.max_seq_length, 300, 300, 1], dtype=np.uint8)
            seq_x = np.zeros([self.max_seq_length, *self.target_image_size, 1], dtype=np.uint8)
            seq_y = np.zeros([self.max_seq_length, 7], dtype=np.float32)

            #print(self.batch_idx[i], len(self.SequenceLength))

            if self.batch_seq_start_idx[i]+self.max_seq_length < self.batch_seq_end_idx[i]:
                seq_start_idx = self.batch_seq_start_idx[i]
                seq_end_idx = self.batch_seq_start_idx[i]+self.max_seq_length

                is_feed_state.append(True)
                self.batch_seq_start_idx[i] += self.max_seq_length
            else:
                seq_start_idx = self.batch_seq_start_idx[i]
                seq_end_idx = self.batch_seq_end_idx[i]

                is_feed_state.append(False)
                self.batch_seq_start_idx[i] = None

            seq_len = seq_end_idx - seq_start_idx

            seq_imgs = []
            seq_targets = []

            for img, target in zip(self.Images[seq_start_idx:seq_end_idx], self.Targets[seq_start_idx:seq_end_idx]):
                #img_argumented, target_argumented = image_argumentation(img, target, target_image_size=[300, 300], mode='train')
                img_argumented, target_argumented = image_argumentation(img, target, target_image_size=self.target_image_size, mode='train')
                seq_imgs.append(img_argumented)
                seq_targets.append(target_argumented)

            seq_x[:seq_len, ...] = seq_imgs
            seq_y[:seq_len, ...] = seq_targets

            batch_x.append(seq_x)
            batch_y.append(seq_y)
            batch_seq_len.append(seq_len)

        return batch_x, batch_y, batch_seq_len, is_feed_state




class PCDataLoader:
    def __init__(self, batch_size=32,  dataset_dir=''):

        self.batch_size = batch_size
        self.dataset_dir = dataset_dir

        text_file = open(os.path.join(dataset_dir, 'poses.txt'), "r")
        self.poses = text_file.readlines()
        self.num_inst = len(self.poses)
        self.num_batches = len(self.poses) / batch_size

    def shuffle_data(self, mode='train'):

        self.idx = np.arange(len(self.poses))
        self.is_epoch_done = False
        self.pointer = 0

        if mode == 'train':
            np.random.shuffle(self.idx)
            self.is_data_argumentation = False
            self.delta_t = 2.0
            self.delta_r = math.pi/180*15


    def get_pose(self, idx):
        line = self.poses[idx]
        tmp2 = line.split(" ")

        tmp = []
        for t in tmp2:
            if len(t):
                tmp.append(t)

        px = float(tmp[1])
        py = float(tmp[2])
        pz = float(tmp[3])

        qx = float(tmp[4])
        qy = float(tmp[5])
        qz = float(tmp[6])
        qw = float(tmp[7][:-2])

        T = tf_tran.quaternion_matrix([qx, qy, qz, qw])
        T[0:3, 3] = [px, py, pz]

        return T


    def next_batch(self, b, mode='train'):

        start_idx = b * self.batch_size
        end_idx = (b + 1) * self.batch_size

        batch_idx = self.idx[start_idx:end_idx]

        batch_x = []
        batch_y = []

        for idx in batch_idx:
            pc_np = np.asarray(pcl.load(os.path.join(self.dataset_dir, 'PCDs', str(idx+1) + '.pcd')))
            T = self.get_pose(idx)

            vels = np.hstack([pc_np, np.ones([pc_np.shape[0], 1])])
            vels = np.matmul(np.linalg.inv(T), vels.transpose())

            if self.is_data_argumentation:

                deltaT = tf_tran.euler_matrix(0, 0, 0, 'rxyz')

                if random.random() < 1:
                    delta_r = (random.random() * 2 - 1) * self.delta_r

                    if random.random() < 0.5:
                        delta_r += math.pi

                    deltaT = tf_tran.euler_matrix(0, 0, delta_r, 'rxyz')

                if random.random() < 1:
                    delta_t = np.array([(random.random()*2-1), (random.random()*2-1), 0.]) * self.delta_t
                    deltaT[0:3, 3] += delta_t

                vels1 = vels.transpose()[:, :3]
                bird_view_img1 = lidar_projection.birds_eye_point_cloud(vels1, side_range=(-50, 50), fwd_range=(-50, 50),
                                                                        res=0.25, min_height=-4, max_height=1)

                vels2 = np.matmul(deltaT, vels)
                vels = vels2
                
                T2 = np.matmul(deltaT, T)
                T = T2

            vels = vels.transpose()[:, :3]
            bird_view_img = lidar_projection.birds_eye_point_cloud(vels, side_range=(-50, 50), fwd_range=(-50, 50), res=0.25, min_height=-4, max_height=1)

            position = np.array(tf_tran.translation_from_matrix(T), dtype=np.single)
            euler = np.array(tf_tran.euler_from_matrix(T, 'rxyz'), dtype=np.single)

            #import cv2
            #cv2.imwrite("tmp1.png", bird_view_img1)
            #cv2.imwrite("tmp2.png", bird_view_img)

            batch_x.append(bird_view_img[:, :, np.newaxis])
            batch_y.append(np.concatenate((position, euler), axis=0))

        return batch_x, batch_y


def main():

    d = DataLoader(batch_size=1, dataset_dir='/home/kevin/data/michigan_gt/exp1')

    d.shuffle_data(mode='val')

    [x, y] = d.next_batch(1, mode='val')


    map_files = ['parking_lot_map_'+str(i)+'_res_40cm.pcd' for i in range(1, 2)]
    bbox_feat_files = ['bbox_feat_'+str(i)+'.txt' for i in range(1, 2)]
    annotation_files = ['parking_lot_map_'+str(i)+'.txt' for i in range(1, 2)]


    [PC, Bbox_Feat, Annotation] = load_all_data(map_dir, map_files, bbox_feat_dir, bbox_feat_files, annotation_dir, annotation_files)

    recurrent_map = convert_bbox_to_point_wise(PC, Bbox_Feat, Annotation)

if __name__=="__main__":main()