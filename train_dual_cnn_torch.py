#!/usr/bin/env python3
# coding: utf-8

# # Torch
# ## Check GPU

# In[1]:


#from apex import amp,optimizers


# In[2]:


import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(1))


# ## Set torch default parameters

# In[3]:


torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)
torch.backends.cudnn.benchmark = True


# # Set Arguments

# In[4]:


import argparse
import sys
import os
import time
import pickle

parser = argparse.ArgumentParser()

'''Training Parameters'''
parser.add_argument('--batch_size', type=int, default=32, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
parser.add_argument('--learning_rate_clip', type=float, default=0.0000001, help='learning rate clip')
parser.add_argument('--decay_rate', type=float, default=.95, help='decay rate for rmsprop')
parser.add_argument('--weight_decay', type=float, default=.0001, help='decay rate for rmsprop')
parser.add_argument('--batch_norm_decay', type=float, default=.999, help='decay rate for rmsprop')
parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability')
parser.add_argument('--lamda_weights', type=float, default=0.1, help='lamda weight')
parser.add_argument('--data_argumentation', type=bool, default=True, help='whether do data argument')
parser.add_argument('--is_normalization', type=bool, default=True, help='whether do data nomalization')
parser.add_argument('--target_image_size', default=[300, 300], nargs=2, type=int, help='Input images will be resized to this for data argumentation.')

'''Configure'''
parser.add_argument('--network', type=str, default='vggnet_localization')
parser.add_argument('--model_dir', type=str, default='/notebooks/global_localization/dual_resnet_torch', help='rnn, gru, or lstm')

parser.add_argument('--train_dataset', type=str, default = ['/notebooks/michigan_nn_data/2012_01_08',
                                                            '/notebooks/michigan_nn_data/2012_01_15',
                                                            '/notebooks/michigan_nn_data/2012_01_22',
                                                            '/notebooks/michigan_nn_data/2012_02_02',
                                                            '/notebooks/michigan_nn_data/2012_02_04',
                                                            '/notebooks/michigan_nn_data/2012_02_05',
                                                            '/notebooks/michigan_nn_data/2012_03_31',
                                                            '/notebooks/michigan_nn_data/2012_09_28'])
'''
parser.add_argument('--train_dataset', type=str, default = ['/notebooks/michigan_nn_data/test'])
'''
parser.add_argument('--seed', default=1337, type=int)
parser.add_argument('--save_every', type=int, default=2000, help='save frequency')
parser.add_argument('--display', type=int, default=10, help='display frequency')

sys.argv = ['']
args = parser.parse_args()


# # Load Dataset

# In[5]:


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tf.transformations as tf_tran
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

import torch.nn as nn
import torch.optim as optim
from torchlib import resnet, vggnet
from torchlib.utils import LocalizationDataset
import time


# In[6]:


transform = transforms.Compose([transforms.ToTensor()])
dataset = LocalizationDataset(dataset_dirs = args.train_dataset,                               image_size = args.target_image_size,                               transform = transform)
[args.norm_mean, args.norm_std] = [torch.tensor(x) for x in dataset.get_norm()]

dataloader = DataLoader(dataset, batch_size=args.batch_size,                         shuffle=True, num_workers=0,                         drop_last=True, pin_memory=True)


# # Define Model

# In[7]:


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet.resnet50(pretrained=True) # dense_feat
        self.global_context = vggnet.vggnet(input_channel=2048,opt="context")
        #self.relative_context = vggnet(input_channel=4096,opt="context")
        self.global_regressor = vggnet.vggnet(opt="regressor")
        
    def forward(self, input_data_t0, input_data_t1):
        dense_feat0 = self.resnet(input_data_t0)
        dense_feat1 = self.resnet(input_data_t1)
        #dense_feat_relative = torch.cat([dense_feat0,dense_feat1],dim=1)
        
        global_context_feat0 = self.global_context(dense_feat0)
        global_context_feat1 = self.global_context(dense_feat1)
        #relative_context_feat = self.relative_context(dense_feat_relative)
        
        global_output0,_,_ = self.global_regressor(global_context_feat0)
        global_output1,_,_ = self.global_regressor(global_context_feat1)
        
        return global_output0,global_output1#,relative_context_feat 
        
def quanternion2matrix(q):
    tx, ty, tz, qx, qy, qz, qw = torch.split(q,[1, 1, 1, 1, 1, 1, 1], dim=-1)
    M11 = 1.0 - 2 * (torch.square(qy) + torch.square(qz))
    M12 = 2. * qx * qy - 2. * qw * qz
    M13 = 2. * qw * qy + 2. * qx * qz
    M14 = tx

    M21 = 2. * qx * qy + 2. * qw * qz
    M22 = 1. - 2. * (torch.square(qx) + torch.square(qz))
    M23 = -2. * qw * qx + 2. * qy * qz
    M24 = ty

    M31 = -2. * qw * qy + 2. * qx * qz
    M32 = 2. * qw * qx + 2. * qy * qz
    M33 = 1. - 2. * (torch.square(qx) + torch.square(qy))
    M34 = tz

    M41 = torch.zeros_like(M11)
    M42 = torch.zeros_like(M11)
    M43 = torch.zeros_like(M11)
    M44 = torch.ones_like(M11)

    #M11.unsqueeze_(-1)
    M11 = torch.unsqueeze(M11, axis=-1)
    M12 = torch.unsqueeze(M12, axis=-1)
    M13 = torch.unsqueeze(M13, axis=-1)
    M14 = torch.unsqueeze(M14, axis=-1)

    M21 = torch.unsqueeze(M21, axis=-1)
    M22 = torch.unsqueeze(M22, axis=-1)
    M23 = torch.unsqueeze(M23, axis=-1)
    M24 = torch.unsqueeze(M24, axis=-1)

    M31 = torch.unsqueeze(M31, axis=-1)
    M32 = torch.unsqueeze(M32, axis=-1)
    M33 = torch.unsqueeze(M33, axis=-1)
    M34 = torch.unsqueeze(M34, axis=-1)

    M41 = torch.unsqueeze(M41, axis=-1)
    M42 = torch.unsqueeze(M42, axis=-1)
    M43 = torch.unsqueeze(M43, axis=-1)
    M44 = torch.unsqueeze(M44, axis=-1)

    M_l1 = torch.cat([M11, M12, M13, M14], axis=2)
    M_l2 = torch.cat([M21, M22, M23, M24], axis=2)
    M_l3 = torch.cat([M31, M32, M33, M34], axis=2)
    M_l4 = torch.cat([M41, M42, M43, M44], axis=2)

    M = torch.cat([M_l1, M_l2, M_l3, M_l4], axis=1)

    return M

def matrix2quternion(M):
    tx = M[:, 0, 3].unsqueeze(-1)
    ty = M[:, 1, 3].unsqueeze(-1)
    tz = M[:, 2, 3].unsqueeze(-1)
    qw = 0.5 * torch.sqrt(M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] + M[:, 3, 3]).unsqueeze(-1)

    mask = torch.abs(qw)<10e-6
    qw = qw if mask.sum()==mask.shape[0] else qw+10e-6

    qx = torch.unsqueeze(M[:, 2, 1] - M[:, 1, 2],-1) / (4. * qw)
    qy = torch.unsqueeze(M[:, 0, 2] - M[:, 2, 0],-1) / (4. * qw)
    qz = torch.unsqueeze(M[:, 1, 0] - M[:, 0, 1],-1) / (4. * qw)
    q = torch.cat([tx, ty, tz, qx, qy, qz, qw], dim=-1)
    return q

def get_relative_pose(Q_a,Q_b):
    M_a = quanternion2matrix(Q_a)
    M_b = quanternion2matrix(Q_b)

    try:
        M_delta = torch.matmul(M_a.inverse(),M_b)
    except ValueError:
        print("matrix is not invertiable")
        M_delta = torch.eye(4).repeat(M_a.shape[0],1,1)

    Q_delta = matrix2quternion(M_delta)

    return Q_delta

def normalize(target, norm_mean, norm_std):
    target_trans = target[:,:3]
    target_trans = torch.div(torch.sub(target_trans,norm_mean),norm_std)
    target_normed = torch.cat([target_trans,target[:,3:]],dim=1)
    return target_normed 

def translational_rotational_loss(pred=None, gt=None, lamda=None):
    trans_pred, rot_pred = torch.split(pred, [3,4], dim=1)
    trans_gt, rot_gt = torch.split(gt, [3, 4], dim=1)
    
    trans_loss = nn.functional.mse_loss(input=trans_pred, target=trans_gt)
    rot_loss = 1. - torch.mean(torch.square(torch.sum(torch.mul(rot_pred,rot_gt),dim=1)))
    
    loss = trans_loss + lamda * rot_loss

    return loss#, trans_loss, rot_loss

def _quanternion2matrix(q):
    tx, ty, tz, qx, qy, qz, qw = torch.split(q,[1, 1, 1, 1, 1, 1, 1], dim=-1)
    M11 = 1.0 - 2 * (torch.square(qy) + torch.square(qz))
    M12 = 2. * qx * qy - 2. * qw * qz
    M13 = 2. * qw * qy + 2. * qx * qz
    M14 = tx

    M21 = 2. * qx * qy + 2. * qw * qz
    M22 = 1. - 2. * (torch.square(qx) + torch.square(qz))
    M23 = -2. * qw * qx + 2. * qy * qz
    M24 = ty

    M31 = -2. * qw * qy + 2. * qx * qz
    M32 = 2. * qw * qx + 2. * qy * qz
    M33 = 1. - 2. * (torch.square(qx) + torch.square(qy))
    M34 = tz

    M41 = torch.zeros_like(M11)
    M42 = torch.zeros_like(M11)
    M43 = torch.zeros_like(M11)
    M44 = torch.ones_like(M11)

    #M11.unsqueeze_(-1)
    M11.unsqueeze_(axis=-1)
    M12.unsqueeze_(axis=-1)
    M13.unsqueeze_(axis=-1)
    M14.unsqueeze_(axis=-1)

    M21.unsqueeze_(axis=-1)
    M22.unsqueeze_(axis=-1)
    M23.unsqueeze_(axis=-1)
    M24.unsqueeze_(axis=-1)

    M31.unsqueeze_(axis=-1)
    M32.unsqueeze_(axis=-1)
    M33.unsqueeze_(axis=-1)
    M34.unsqueeze_(axis=-1)

    M41.unsqueeze_(axis=-1)
    M42.unsqueeze_(axis=-1)
    M43.unsqueeze_(axis=-1)
    M44.unsqueeze_(axis=-1)

    M_l1 = torch.cat([M11, M12, M13, M14], axis=2)
    M_l2 = torch.cat([M21, M22, M23, M24], axis=2)
    M_l3 = torch.cat([M31, M32, M33, M34], axis=2)
    M_l4 = torch.cat([M41, M42, M43, M44], axis=2)

    M = torch.cat([M_l1, M_l2, M_l3, M_l4], axis=1)

    return M

def _matrix2quternion(M):
    tx = M[:, 0, 3].unsqueeze_(-1)
    ty = M[:, 1, 3].unsqueeze_(-1)
    tz = M[:, 2, 3].unsqueeze_(-1)
    qw = 0.5 * torch.sqrt(M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] + M[:, 3, 3]).unsqueeze_(-1)

    mask = torch.abs(qw)<10e-6
    qw = qw if mask.sum()==mask.shape[0] else qw+10e-6

    qx = torch.unsqueeze(M[:, 2, 1] - M[:, 1, 2],-1) / (4. * qw)
    qy = torch.unsqueeze(M[:, 0, 2] - M[:, 2, 0],-1) / (4. * qw)
    qz = torch.unsqueeze(M[:, 1, 0] - M[:, 0, 1],-1) / (4. * qw)
    q = torch.cat([tx, ty, tz, qx, qy, qz, qw], dim=-1)
    return q


# In[8]:


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

# set to cpu
#device = torch.device("cpu")
net = Model().to(device)


# ## Model Structure

# In[9]:


for name, param in net.named_parameters():
    if param.requires_grad:
        print (name, param.shape)


# # Training
# ## Parameters

# In[10]:


args.norm_mean = args.norm_mean.to(device)
args.norm_std = args.norm_std.to(device)

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
#optimizer = optimizers.FusedAdam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: args.decay_rate**epoch)

#net, optimizer = amp.initialize(net, optimizer, opt_level="O1")


# ## Training Epoch

# In[11]:


for e in range(args.num_epochs):
#for e in range(2):
    if e != 0:
        scheduler.step()
    train_loss = 0.
    for b, data in enumerate(dataloader, 0):

        start = time.time()
        optimizer.zero_grad()
        
        x0, x1 = data['image']
        y0, y1 = data['target']
        x0,x1,y0,y1 = x0.to(device),x1.to(device),y0.to(device),y1.to(device)
        # normalize targets
        y0_norm, y1_norm = [normalize(y,args.norm_mean, args.norm_std) for y in [y0,y1]]
        relative_target_normed = get_relative_pose(y0_norm, y1_norm)
        
        # Part 1: Net Forward
        global_output0,global_output1 = net(x0, x1)

        # Part 2: Loss
        
        relative_consistence = get_relative_pose(global_output0,global_output1)

        global_loss = translational_rotational_loss(pred=global_output1,                                                     gt=y1_norm,                                                     lamda=args.lamda_weights)
        geometry_consistent_loss = translational_rotational_loss(pred=relative_consistence,                                                                  gt=relative_target_normed,                                                                  lamda=args.lamda_weights)
        total_loss = global_loss + 0.1 * geometry_consistent_loss
        
        # Part 3: Net Backward
        #with amp.scale_loss(total_loss, optimizer) as scaled_loss:
        #    scaled_loss.backward()
        
        total_loss.backward()
        optimizer.step()

        #del global_output0,global_output1,relative_consistence,global_loss,geometry_consistent_loss
        end = time.time()

        with torch.no_grad():
            train_loss += float(total_loss)
            if ((b+1)%args.display == 0):
                 print(
                    "{}/{} (epoch {}), train_loss = {}, time/batch = {:.3f}, learning rate = {:.9f}"
                    .format(
                    e * len(dataloader) + (b+1),
                    args.num_epochs * len(dataloader),
                    e,
                    train_loss/(b+1),
                    end - start,
                    optimizer.param_groups[0]['lr']))
            if (e * len(dataloader) + (b+1)) % args.save_every == 0:
                checkpoint_path = os.path.join(args.model_dir, 'model-{}-{}.pth'.format(e, e * len(dataloader) + (b+1)))
                torch.save(net.state_dict(),checkpoint_path)
                print('saving model to model-{}-{}.pth'.format(e, e * len(dataloader) + (b+1)))


# In[ ]:




