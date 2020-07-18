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
import torch
from torch.utils.data import Dataset

def list_device():
    if torch.cuda.is_available():
        print('------------ List Devices ------------')
        for i in range(torch.cuda.device_count()):
            print('Device',i,':')
            print(torch.cuda.get_device_name(i))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(i)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(i)/1024**3,1), 'GB\n')
            
def set_device(i):
    device = torch.device("cuda:"+str(i) if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        i = torch.cuda.current_device()
        print('Using Device',i,':',torch.cuda.get_device_name(i))

class LocalizationDataset(Dataset):
    """class:`LocalizationDataset`

    This class is used to load data. Designed for pytorch's DataLoader.
    Get a dictionary of data for each iteration, which contains 2 lists
    of data pairs.
    E.g. {'image': [image_0, image_1], 'target': [target_0, target_1]}

    Attributes:
        dataset_dirs->list: A list of the path of the dataset. Each path
            should contain at least the index.txt file and the images 
            and poses folders.
        image_size->list: Specify the shape of image, e.g. [300,300].
        sampling_rate->int: The sample rate. Must be greater than 0, 
            e.g. 5
        frames->int: The number of relative frames. Must be greater than 0.
        transform->torchvision.transforms: Image transform only.
        normalize->bool: Whether the target needs to be normalized.
        
    Raises:
        TypeError: `sampling_rate` and `frames` should be integer.
    """
    def __init__(self, dataset_dirs, image_size=[300,300], frames=10, sampling_rate=2, \
                 transform=None, normalize=False, get_pair = True, mode = 'train'):
        self.dataset_dirs = dataset_dirs
        self.image_size = image_size
        self.num_connected_frames = frames
        self.sample = sampling_rate
        self.transform = transform
        self.normalize = normalize
        self.get_pair = get_pair
        
        self.Images = list()
        self.Targets = list()
        
        self.__loaddata(mode)
        self.norm_mean,self.norm_std = self.get_norm()
        
    def __loaddata(self,mode='train'):
        for dataset_dir in self.dataset_dirs:
            # Read index from files
            with open(os.path.join(dataset_dir, 'index.txt'), "r") as f:
                file_index = f.readlines() # a list of indexes
                file_index = file_index[self.sample-1::self.sample]

            for index in tqdm(file_index):
                # Read images
                img = np.array(Image.open(os.path.join(dataset_dir, 'images', index[:-1] + '.png')), dtype=np.uint8)
                img = img[:, :, np.newaxis]
                # Read poses
                target = np.loadtxt(os.path.join(dataset_dir, 'poses', index[:-1] + '.txt'))
                [px, py, pz, ex, ey, ez] = target
                [qx, qy, qz, qw] = tf_tran.quaternion_from_euler(ex, ey, ez, 'rxyz')
                target = np.array([px, py, pz, qx, qy, qz, qw], dtype=np.float32)
                #np.testing.assert_approx_equal(np.sum(target[-4:] ** 2), 1.0, significant=5)
                
                img,target = self._image_argumentation(img,target,self.image_size,mode=mode)
                
                self.Images.append(img)
                self.Targets.append(target)
    
    def _normalize(self, target):
        target_trans = target[:3]
        target_trans = (target_trans-self.norm_mean)/self.norm_std
        target_normed = np.hstack([target_trans,target[3:]])
        return target_normed 
    
    def _image_argumentation(self, img, target, target_image_size=[200, 200], mode='train', rot_angle=None):
        RES = 100.0 / 400.0

        margin_row = img.shape[0] - target_image_size[0]
        margin_col = img.shape[1] - target_image_size[1]

        [px, py, pz, qx, qy, qz, qw] = target

        if mode == 'train':
            #np.random.seed(0) # fixed
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
    
    def __len__(self):
        if self.get_pair:
            return len(self.Targets) - self.num_connected_frames
        else:
            return len(self.Targets)

    def __getitem__(self, idx): # next_pair
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.get_pair:
            idx +=  self.num_connected_frames
        
        image_1 = self.Images[idx]
        target_1 = self.Targets[idx]
        if self.get_pair:
            #random.seed(0) # fixed
            paired_frame_offset = random.randint(1, self.num_connected_frames)
            image_0 = self.Images[idx-paired_frame_offset]
            target_0 = self.Targets[idx-paired_frame_offset]
        
        
        # move to `__loaddata()` 
        #image_0, target_0 = self._image_argumentation(image_0,target_0,self.image_size)
        #image_1, target_1 = self._image_argumentation(image_1,target_1,self.image_size)
        
        if self.normalize:
            target_1 = self._normalize(target_1)
            if self.get_pair:
                target_0 = self._normalize(target_0)
            
        
        #np.testing.assert_approx_equal(np.sum(target_0[-4:] ** 2), 1.0, significant=5)
        #np.testing.assert_approx_equal(np.sum(target_1[-4:] ** 2), 1.0, significant=5)
        
        if self.transform:
            image_1 = self.transform(image_1)
            if self.get_pair:
                image_0 = self.transform(image_0)
            
        if self.get_pair:  
            sample = {'image': [image_0, image_1], 'target': [target_0, target_1]}
        else:
            sample = {'image': image_1, 'target': target_1}
        return sample
    
    def get_norm(self):
        ''' get the mean and std of pose [px, py, pz]
        Args:
            pos->np.array: [[px, py, pz],...,[px, py, pz]], with shape of (m, 3)
        Returns:
            norm_mean->np.array: [x_mean, y_mean, z_mean], with shape of (3,)
            norm_std->np.array: [x_std, y_std, z_std], with shape of (3,)
        '''
        pos = np.array(self.Targets)[:,:3]
        norm_mean = np.mean(pos, axis=0)
        norm_std = np.std(pos - norm_mean,axis=0)
        return norm_mean,norm_std
    
def normalize(target, norm_mean, norm_std):
    target_trans = target[:,:3]
    target_trans = torch.div(torch.sub(target_trans,norm_mean),norm_std)
    target_normed = torch.cat([target_trans,target[:,3:]],dim=1)
    return target_normed 
    
def display_loss(present_step,total_step,epoch,train_loss,batch_time,lr):
    print(
        "{}/{} (epoch {}), train_loss = {:.8f}, time/batch = {:.3f}, learning rate = {:.8f}"
        .format(present_step,total_step,epoch,train_loss,batch_time,lr))
    
def data2tensorboard(writer,item_loss,batch_loss,present_step):
    writer.add_scalars('training loss',
                  {'item loss':item_loss,
                  'batch loss':batch_loss},
                  present_step)