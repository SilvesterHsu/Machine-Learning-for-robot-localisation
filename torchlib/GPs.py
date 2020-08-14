import torch
import torch.nn as nn
from torch.distributions import Normal
import os
import sys
sys.path.append('..')

from torchlib import resnet, vggnet
from torchlib.cnn_auxiliary import normalize, denormalize, denormalize_navie, get_relative_pose, translational_rotational_loss
import gpytorch

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet.resnet50(pretrained=True)
    def forward(self,input_data):
        dense_feat = self.resnet(input_data)
        return dense_feat
    
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_context = vggnet.vggnet(input_channel=2048,opt="context")
        self.global_regressor = vggnet.vggnet(opt="regressor")
        
    def forward(self,input_data):
        context_feat = self.global_context(input_data)
        output,feature_t, feature_r = self.global_regressor(context_feat)
        return output, feature_t, feature_r

class GP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, output_dim=3):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([output_dim])
        )
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ), num_tasks=output_dim
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dim]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([output_dim])),
            batch_shape=torch.Size([output_dim]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPNode(nn.Module):
    def __init__(self, inducing_points, seed=0, feat_dim = 128, sub_feat = True):
        super().__init__()
        output_dim = inducing_points.shape[0]
        
        if sub_feat:
            sub_feat_dim = inducing_points.shape[-1]
            torch.manual_seed(seed)
            self.feat_index = torch.randperm(feat_dim)[:sub_feat_dim]
        self.gp = GP(inducing_points,output_dim)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim) 
        
    def forward(self,input_data):
        output = self.gp(input_data)
        return output
    
class Model(nn.Module):
    def __init__(self, num_gp = 20, sub_feat_rate = 0.6666, feat_dim = 128, output_dim = 3):
        super().__init__()
        self.backbone = Backbone()
        self.nn = NN()
        self.gps = nn.ModuleList()
        
        self.num_gp = num_gp
        self.sub_feat_rate = sub_feat_rate
        self.sub_feat_dim = int(feat_dim*self.sub_feat_rate)
        
        for i in range(self.num_gp):
            inducing_points = torch.zeros(output_dim, 300, self.sub_feat_dim)
            # use i as seed to fix sub features
            gp = GPNode(inducing_points,seed=i)
            self.gps.append(gp)
        
    def forward_nn(self, input_data):
        dense_feat = self.backbone(input_data)
        output, feature_t, feature_r = self.nn(dense_feat)
        rot_pred = torch.split(output, [3, 4], dim=1)[1] # 4-dimention            
        return feature_t, rot_pred
    
    def forward_gp(self,gp,trans_feat):
        sub_trans_feat = trans_feat[:,gp.feat_index]
        trans_pred = gp(sub_trans_feat)
        return trans_pred

class BaseModule:
    def __init__(self, norm_mean, norm_std, args):
        self.model = None
        self.args = args
        self.device = torch.device("cuda:"+str(args.cuda_device) if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(args.cuda_device)
        self.norm_mean = norm_mean.to(self.device)
        self.norm_std = norm_std.to(self.device)
        
    def disable_requires_grad(self,model):
        for param in model.parameters():
            param.requires_grad = False
            
    def load_model(self, file_name = 'pretrained.pth', strict = True):
        # load file info
        state_dict = torch.load(os.path.join(self.args.model_dir, file_name))
        if 'net.resnet.conv1.weight' in state_dict:
            print('Transform from old model.')
            state_dict = self._from_old_model(state_dict)
            self.model.load_state_dict(state_dict,strict = strict)
        else:
            #print('Parameters layer:',len(state_dict.keys()))
            # load file to model
            self.model.load_state_dict(state_dict,strict = strict)
        print("Successfully loaded model to {}.".format(self.device_name))
    
    def _from_old_model(self, state_dict, select = 'backbone'):
        if select == 'backbone':
            for key in list(state_dict):
                if 'net.resnet.' in key:
                    state_dict[key.replace('net.resnet.','resnet.')] = state_dict.pop(key)
                else:
                    state_dict.pop(key)
        elif select == 'nn':
            for key in list(state_dict):
                if 'net.global_regressor.' in key:
                    state_dict[key.replace('net.global_regressor.','global_regressor.')] = state_dict.pop(key)
                elif 'net.global_context.' in key:
                    state_dict[key.replace('net.global_context.','global_context.')] = state_dict.pop(key)
                else:
                    state_dict.pop(key)
        return state_dict
    
    def save_model(self, file_name = 'model-{}-{}.pth'):
        checkpoint_path = os.path.join(self.args.model_dir, file_name)
        torch.save(self.model.state_dict(),checkpoint_path)
        print('Saving model to ' +  file_name)
        
    def show_require_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print (name, param.shape)
                
    def schedule_step(self,step,step_every = 150):
        if self.scheduler.get_last_lr()[-1] > self.args.learning_rate_clip and step % step_every == 0:
            self.scheduler.step()
            
    def save_model_step(self,e,step):
        if step % self.args.save_every == 0:
            self.save_model('model-{}-{}.pth'.format(e, step))
            
    def data2tensorboard(self,writer,name,data,step):
        if self.args.tensorboard:
            writer.add_scalars(name,data,step)

class PosePredictor(BaseModule):
    def __init__(self, norm_mean, norm_std, args,
                 regressor_context_rate = [0.0,0.0],
                 is_training=True, disable_rot_learning = True):
        
        super().__init__(norm_mean, norm_std, args)
        self.model = Model().to(self.device)
        
        # disable learning backbone
        self.disable_requires_grad(self.model.backbone)
        
        if is_training:
            # training tool
            self.optimizer = optim.Adam(self._optimize(regressor_context_rate))
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                                             lr_lambda=lambda epoch: args.decay_rate**epoch)
        else:
            self.disable_requires_grad(self.model)
            
    def _optimize(self, regressor_context_rate = [0.0,0.0]):
        optimizer = [
                {'params': self.model.gps.parameters(), \
                 'lr': self.args.learning_rate,'weight_decay':self.args.weight_decay}]
            
        if regressor_context_rate[0]!=0:
            optimizer += [{'params': self.model.nn.global_regressor.parameters(), \
                 'lr': self.args.learning_rate * regressor_context_rate[0],'weight_decay':self.args.weight_decay}]
            print('Regressor learn rate:',regressor_context_rate[0])
        else:
            for param in self.model.nn.global_regressor.parameters():
                param.requires_grad = False
                
        if regressor_context_rate[1]!=0:
            optimizer += [{'params': self.model.nn.global_context.parameters(), \
                 'lr': self.args.learning_rate * regressor_context_rate[1],'weight_decay':self.args.weight_decay}]
            print('Context learn rate:',regressor_context_rate[1])
        else:
            for param in self.model.nn.global_context.parameters():
                param.requires_grad = False
                
        return optimizer
    
    def train(self, x, y):
        # Step 0: zero grad
        self.optimizer.zero_grad()
        
        start = time.time()
        # Step 1: get data
        x,y = x.to(self.device),y.to(self.device)
        if self.args.is_normalization:
            y = normalize(y,self.norm_mean, self.norm_std)
            
        # Step 2: training
        assert self.model.training == True
        
        trans_loss = torch.tensor(0.).to(self.device)
        
        trans_target, rot_target = torch.split(y, [3, 4], dim=1)
        trans_feat, rot_pred = self.model.forward_nn(x)
        rot_loss = self._nn_loss(rot_pred,rot_target)
        for i,gp in enumerate(self.model.gps):
            #torch.manual_seed(i)
            #sampled_mask = torch.randint(high=args.batch_size, size=(self.model.sub_batch_size,))
            sampled_mask = torch.randint(high=self.args.batch_size, size=(self.args.batch_size,))
            sub_x = trans_feat[sampled_mask]
            sub_y = trans_target[sampled_mask]
            gp_loss = self._gp_loss(gp,sub_x,sub_y)
            trans_loss += gp_loss
        trans_loss = trans_loss/self.model.num_gp
        
        total_loss = trans_loss + self.args.lamda_weights * rot_loss
        
        batch_time = time.time() - start
        
        #Step 3: update
        total_loss.backward()
        self.optimizer.step()
        
        return float(total_loss), batch_time    
    
    def _nn_loss(self, rot_pred, rot_target):
        rot_loss = 1. - torch.mean(torch.square(torch.sum(torch.mul(rot_pred,rot_target),dim=1)))
        return rot_loss
        
    def _gp_loss(self, gp, trans_feat, trans_target):
        # predict
        trans_pred = self.model.forward_gp(gp,trans_feat)
        
        #num_data = int(min(len(dataloader)*args.batch_size,len(dataset))*self.model.sub_batch_rate)
        num_data = min(len(dataloader)*self.args.batch_size,len(dataset))
        mll = gpytorch.mlls.PredictiveLogLikelihood(gp.likelihood, gp.gp, num_data = num_data)
        
        # trans loss
        trans_loss = -1.*mll(trans_pred, trans_target)
        
        return trans_loss
    
    def _eval_gp(self, gp, trans_pred):
        c_mean, c_var = trans_pred.mean, trans_pred.variance
        y_mean, y_var = gp.likelihood(trans_pred).mean, gp.likelihood(trans_pred).variance
        
        return y_mean, c_mean, c_var
    
    def _sample(self, mean, var, num_sample = 100):
        dist = Normal(mean, var)
        samples = dist.sample([num_sample])
        return samples

    def eval_forward(self, x, num_sample = 100, output_denormalize = True):
        # Step 1: get data
        x = x.to(self.device)
        
        # Step 2: forward
        assert self.model.training == False
        trans_feat, rot_pred = self.model.forward_nn(x)
        
        trans_preds = 0
        trans_means = 0
        trans_vars = 0
        for gp in self.model.gps:
            trans_pred = self.model.forward_gp(gp,trans_feat)
            trans_pred, trans_mean, trans_var = self._eval_gp(gp, trans_pred)
            trans_preds += trans_pred * 1/trans_var
            trans_means += trans_mean * 1/trans_var
            trans_vars += 1/trans_var
         
        trans_vars = 1/trans_vars
        trans_preds *= trans_vars
        trans_means *= trans_vars
        
        if self.args.is_normalization and output_denormalize:
            trans_preds = denormalize_navie(trans_preds, self.norm_mean, self.norm_std)
            trans_means = denormalize_navie(trans_means, self.norm_mean, self.norm_std)
            trans_vars = trans_vars.mul(self.norm_std)
        
        samples = self._sample(trans_means, trans_vars, num_sample)
            
        
        return trans_preds, rot_pred, trans_means, trans_vars, samples