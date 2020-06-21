#!/usr/bin/env python
# coding: utf-8

# # Torch
# ## Check GPU¶

# In[1]:


#from apex import amp,optimizers


# In[2]:


import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(1))


# ## Set torch default parameters¶

# In[3]:


torch.set_default_dtype(torch.float32)
torch.set_printoptions(precision=3)
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
parser.add_argument('--batch_size', type=int, default=300, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.002, help='learning rate')
parser.add_argument('--learning_rate_clip', type=float, default=0.0000001, help='learning rate clip')
parser.add_argument('--decay_rate', type=float, default=.9, help='decay rate for rmsprop')
parser.add_argument('--weight_decay', type=float, default=.0001, help='decay rate for rmsprop')
parser.add_argument('--batch_norm_decay', type=float, default=.999, help='decay rate for rmsprop')
parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability')
parser.add_argument('--lamda_weights', type=float, default=10, help='lamda weight')
parser.add_argument('--data_argumentation', type=bool, default=True, help='whether do data argument')
parser.add_argument('--is_normalization', type=bool, default=True, help='whether do data nomalization')
parser.add_argument('--target_image_size', default=[300, 300], nargs=2, type=int, help='Input images will be resized to this for data argumentation.')
parser.add_argument('--output_dim', default=3, type=int, help='output dimention.')
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimention.')

'''Configure'''
parser.add_argument('--network', type=str, default='vggnet_localization')
parser.add_argument('--model_dir', type=str, default='/notebooks/global_localization/gp_net_torch', help='rnn, gru, or lstm')

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
parser.add_argument('--save_every', type=int, default=2500, help='save frequency')
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

#import gpflow.multioutput.kernels as mk
import gpytorch

import torch.nn as nn
import torch.optim as optim
from torchlib import resnet, vggnet
from torchlib.utils import LocalizationDataset
import time

transform = transforms.Compose([transforms.ToTensor()])
dataset = LocalizationDataset(dataset_dirs = args.train_dataset,                               image_size = args.target_image_size,                               transform = transform,
                              get_pair = False)
[args.norm_mean, args.norm_std] = [torch.tensor(x) for x in dataset.get_norm()]

dataloader = DataLoader(dataset, batch_size=args.batch_size,                         shuffle=True, num_workers=0,                         drop_last=True, pin_memory=True)


# # Define Model

# In[6]:


def normalize(target, norm_mean, norm_std):
    target_trans = target[:,:3]
    target_trans = torch.div(torch.sub(target_trans,norm_mean),norm_std)
    target_normed = torch.cat([target_trans,target[:,3:]],dim=1)
    return target_normed 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet.resnet50(pretrained=True)
        self.global_context = vggnet.vggnet(input_channel=2048,opt="context")
        self.global_regressor = vggnet.vggnet(opt="regressor")
        
    def forward(self,input_data):
        dense_feat = self.resnet(input_data)
        global_context_feat = self.global_context(dense_feat)
        global_output, trans_feat, rot_feat = self.global_regressor(global_context_feat)
        return global_output, trans_feat, rot_feat
    
class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([3])
        )

        # We have to wrap the VariationalStrategy in a MultitaskVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ), num_tasks=3
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        #self.net = Model()
        #self.net.load_state_dict(torch.load(os.path.join(args.model_dir,'model-23-96000.pth')))
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([1])),
            batch_shape=torch.Size([1])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class GPModel(gpytorch.Module):
    def __init__(self, inducing_points):
        super(GPModel, self).__init__()
        self.net = Model()
        #self.net.load_state_dict(torch.load(os.path.join('/notebooks/global_localization/dual_resnet_torch','model-23-96000.pth')))
        self.gp = MultitaskGPModel(inducing_points)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)

    def forward(self, x):
        global_output, trans_feat, _ = self.net(x)
        _, rot_pred = torch.split(global_output, [3, 4], dim=1)
        output = self.gp(trans_feat)
        
        return output,rot_pred


# In[7]:


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)

model = GPModel(torch.zeros(3, args.batch_size, 128)).to(device)
#model.net.load_state_dict(torch.load(os.path.join('/notebooks/global_localization/dual_resnet_torch','model-23-96000.pth')))
model.load_state_dict(torch.load(os.path.join(args.model_dir,'pretrained_gp.pth')))

# Disable resnet
for param in model.net.resnet.parameters():
    param.requires_grad = False


# In[8]:


for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.shape)


# # Training
# ## Parameters

# In[9]:


args.norm_mean = args.norm_mean.to(device)
args.norm_std = args.norm_std.to(device)

optimizer = optim.Adam([
    {'params': model.gp.parameters(), \
     'lr': args.learning_rate,'weight_decay':args.weight_decay},
    {'params': model.likelihood.parameters(), \
     'lr': args.learning_rate,'weight_decay':args.weight_decay},
    {'params': model.net.global_context.parameters(), \
     'lr': args.learning_rate * 0.01,'weight_decay':args.weight_decay},
    {'params': model.net.global_regressor.parameters(), \
     'lr': args.learning_rate * 0.001,'weight_decay':args.weight_decay},
])

#scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: args.decay_rate**epoch)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma = args.decay_rate)
#mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp, num_data=len(dataset.Targets))
mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model.gp, num_data=len(dataset.Targets))

# In[10]:


print('CNN model parameters:', sum(param.numel() for param in model.net.global_context.parameters()))
print('Regressor model parameters:', sum(param.numel() for param in model.net.global_regressor.parameters()))
print('GP model parameters:', sum(param.numel() for param in model.gp.parameters()))
print('Likelihood parameters:', sum(param.numel() for param in model.likelihood.parameters()))


# ## Training Epoch

# In[10]:


def train(e):
    train_loss = 0.
    with gpytorch.settings.num_likelihood_samples(8):
        for b, data in enumerate(dataloader, 0):
            start = time.time()
            with torch.no_grad():
                x,y = data.values()
                x,y = x.to(device),y.to(device)
                # normalize targets
                y = normalize(y,args.norm_mean, args.norm_std)
                trans_target, rot_target = torch.split(y, [3, 4], dim=1)
                
            optimizer.zero_grad()
            output,rot_pred = model(x)
            trans_loss = -mll(output, trans_target)
            rot_loss = 1. - torch.mean(torch.square(torch.sum(torch.mul(rot_pred,rot_target),dim=1)))
            total_loss = trans_loss + args.lamda_weights * rot_loss
            total_loss.backward()
            optimizer.step()
            
            end = time.time()
            with torch.no_grad():
                train_loss += float(total_loss)
                lr = scheduler.get_last_lr()[0]
                if ((b+1)%args.display == 0):
                     print(
                        "{}/{} (epoch {}), train_loss = {}, time/batch = {:.3f}, learning rate = {:.9f}"
                        .format(
                        e * len(dataloader) + (b+1),
                        args.num_epochs * len(dataloader),
                        e,
                        train_loss/(b+1),
                        end - start,
                        lr)) # scheduler.get_last_lr()[0]
                if (e * len(dataloader) + (b+1)) % args.save_every == 0:
                    checkpoint_path = os.path.join(args.model_dir, 'model-{}-{}.pth'.format(e, e * len(dataloader) + (b+1)))
                    torch.save(model.state_dict(),checkpoint_path)
                    print('saving model to model-{}-{}.pth'.format(e, e * len(dataloader) + (b+1)))


# In[11]:

model.train()
for epoch in range(args.num_epochs):
    train(epoch)
    scheduler.step()


# In[ ]:




