import torch
import torch.nn as nn
import sys
sys.path.append('..')

from torchlib import resnet, vggnet
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

class GPNode_local(nn.Module):
    def __init__(self,inducing_points):
        super(GPNode_local,self).__init__()
        output_dim = inducing_points.shape[0]
        feat_dim = inducing_points.shape[-1]
        #assert output_dim == args.output_dim
        #assert feat_dim == args.feat_dim
        
        self.gp = GP(inducing_points)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim) 
        
    def forward(self,input_data):
        output = self.gp(input_data)
        return output