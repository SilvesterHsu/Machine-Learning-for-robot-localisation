#!/usr/bin/env python
# coding: utf-8

# # Torch
# ## Check GPU

# In[2]:


import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(1))


# ## Set torch default parameters

# In[3]:


torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=8)
torch.backends.cudnn.benchmark = True

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/train_dual_cnn_torch')


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
parser.add_argument('--learning_rate', type=float, default=0.00004, help='learning rate')
parser.add_argument('--learning_rate_clip', type=float, default=0.0000001, help='learning rate clip')
parser.add_argument('--decay_rate', type=float, default=.7, help='decay rate for rmsprop')
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

# In[6]:


from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import tf.transformations as tf_tran
from tqdm import tqdm
from PIL import Image
import numpy as np
import random

import torch.nn as nn
import torch.optim as optim
from torchlib import resnet, vggnet, cnn_auxiliary
from torchlib.cnn_auxiliary import normalize, denormalize, get_relative_pose, translational_rotational_loss
from torchlib.utils import LocalizationDataset, display_loss, data2tensorboard
import time

transform = transforms.Compose([transforms.ToTensor()])
dataset = LocalizationDataset(dataset_dirs = args.train_dataset,                               image_size = args.target_image_size,                               transform = transform)
[args.norm_mean, args.norm_std] = [torch.tensor(x) for x in dataset.get_norm()]

dataloader = DataLoader(dataset, batch_size=args.batch_size,                         shuffle=True, num_workers=0,                         drop_last=True, pin_memory=True)


# # Define Model

# In[7]:


class CNN_Model:
    def __init__(self, training = True, device = "cpu"):
        # device
        self.device = torch.device(device)
        
        # data
        self.model = cnn_auxiliary.Model(training).to(device)
        self.norm_mean = args.norm_mean.to(device)
        self.norm_std = args.norm_std.to(device)
        
        # training tool
        if training:
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=args.learning_rate, 
                                        weight_decay=args.weight_decay)
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                         lr_lambda=lambda epoch: args.decay_rate**epoch)
        
    def load_model(self, file_name = 'pretrained.pth', display_info = True):
        state_dict = torch.load(os.path.join(args.model_dir, file_name))
        if display_info:
            for name,param in state_dict.items():
                print(name, param.shape)
            print('Parameters layer:',len(state_dict.keys()))
        self.model.load_state_dict(state_dict,strict = False)
        
    def display_structure(self):
        for name, param in self.model.named_parameters():
            print(name, param.shape)
        print('Parameters layer:',len(self.model.state_dict().keys()))
    
    def display_require_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
    
    def power_resnet(self, status = False):
        if status == 'off':
            for param in self.model.resnet.parameters():
                param.requires_grad = False
        elif status == 'on':
            for param in self.model.resnet.parameters():
                param.requires_grad = True
        else:
            raise Exception("status must be 'on' or 'off'.")
            
    def power_context(self, status = False):
        if status == 'off':
            for param in self.model.global_context.parameters():
                param.requires_grad = False
        elif status == 'on':
            for param in self.model.global_context.parameters():
                param.requires_grad = True
        else:
            raise Exception("status must be 'on' or 'off'.")
    
    def power_regressor(self, status = False):
        if status == 'off':
            for param in self.model.global_regressor.parameters():
                param.requires_grad = False
        elif status == 'on':
            for param in self.model.global_regressor.parameters():
                param.requires_grad = True
        else:
            raise Exception("status must be 'on' or 'off'.")
            
    def power_all(self, status = False):
        if status == 'off':
            for param in self.model.parameters():
                param.requires_grad = False
        elif status == 'on':
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise Exception("status must be 'on' or 'off'.")
            
    def save_model(self, file_name = 'model-{}-{}.pth'):
        checkpoint_path = os.path.join(args.model_dir, file_name)
        torch.save(self.model.state_dict(),checkpoint_path)
        print('saving model to ' +  file_name)
            
    def loss(self,x0, x1, y0, y1):
        start = time.time()
        
        x0,x1,y0,y1 = x0.to(self.device),x1.to(self.device),y0.to(self.device),y1.to(self.device)
        #y0_norm, y1_norm = [normalize(y,self.norm_mean, self.norm_std) for y in [y0,y1]]
        y0_norm, y1_norm = y0,y1

        relative_target_normed = get_relative_pose(y0_norm, y1_norm)
        
        #self.optimizer.zero_grad()
        
        global_output0,global_output1 = self.model(x0, x1)
        relative_consistence = get_relative_pose(global_output0,global_output1)
        global_loss = translational_rotational_loss(pred=global_output1,                                                     gt=y1_norm,                                                     lamda=args.lamda_weights)
        geometry_consistent_loss = translational_rotational_loss(pred=relative_consistence,                                                                  gt=relative_target_normed,                                                                  lamda=args.lamda_weights)
        total_loss = global_loss + geometry_consistent_loss        
        #total_loss.backward()
        #self.optimizer.step()
        
        end = time.time()
        batch_time = end - start
        return batch_time, total_loss
    
    def eval_forward(self,x,y):
        x,y = x.to(self.device),y.to(self.device)
        
        global_output = self.model(x)
        trans_target, rot_target = torch.split(y, [3, 4], dim=1)
        global_output_demormed = denormalize(global_output, self.norm_mean, self.norm_std)
        trans_prediction, rot_prediction = torch.split(global_output_demormed, [3, 4], dim=1)
        return trans_prediction, rot_prediction, trans_target, rot_target

cnn_model = CNN_Model(training=True,device="cuda:1")
#cnn_model.load_model('pretrained.pth',display_info=False)
cnn_model.load_model('model-0-4000.pth',display_info=False)


# ## Tensorboard Graphs

# In[8]:


'''
with torch.no_grad():
    graphs = Model()
    x0,x1 = next(iter(dataloader))['image']
    writer.add_graph(graphs, (x0,x1))
del x0,x1,graphs
'''


# ## Model Structure

# In[9]:


cnn_model.display_structure()


# In[10]:


cnn_model.display_require_grad()


# # Training

# ## Training Epoch

# In[11]:


cnn_model.model.train()
for e in range(args.num_epochs):
#for e in range(2):
    train_loss = 0.
    for b, data in enumerate(dataloader, 0):
        x0, x1 = data['image']
        y0, y1 = data['target']
        
        cnn_model.optimizer.zero_grad()
        batch_time, loss = cnn_model.loss(x0,x1,y0,y1)
        loss.backward()
        cnn_model.optimizer.step()
        
        with torch.no_grad():
            train_loss += float(loss)
            data2tensorboard(writer,float(loss),train_loss/(b+1),e*len(dataloader)+(b+1))
            if ((b+1)%args.display == 0):
                 display_loss(e*len(dataloader)+(b+1),args.num_epochs*len(dataloader),e,
                              train_loss/(b+1),batch_time,cnn_model.scheduler.get_last_lr()[0])          
            if (e * len(dataloader) + (b+1)) % args.save_every == 0:
                cnn_model.save_model('model-{}-{}.pth'.format(e, e * len(dataloader) + (b+1)))
            if (e * len(dataloader) + (b+1)) % 2000 == 0:
                cnn_model.scheduler.step()


# In[ ]:




