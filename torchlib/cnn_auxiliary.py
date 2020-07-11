from torchlib import resnet, vggnet
import torch.nn as nn
import torch

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
    eps = torch.finfo(M.dtype).eps
    tx = M[:, 0, 3].unsqueeze(-1)
    ty = M[:, 1, 3].unsqueeze(-1)
    tz = M[:, 2, 3].unsqueeze(-1)
    qw = 0.5 * torch.sqrt(M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] + M[:, 3, 3] + eps).unsqueeze(-1) # sqrt ！= 0 
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

def denormalize_navie(normed_target, norm_mean, norm_std):
    target_trans_unscaled = normed_target * norm_std
    target_trans_uncentered = target_trans_unscaled + norm_mean
    
    return target_trans_uncentered

def denormalize(normed_target, norm_mean, norm_std):
    normed_target_trans, normed_target_rot = torch.split(normed_target, [3,4], dim=1)
    target_trans_unscaled = normed_target_trans * norm_std
    target_trans_uncentered = target_trans_unscaled + norm_mean
    target = torch.cat([target_trans_uncentered, normed_target_rot],dim=1)
    return target

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

class Model(nn.Module):
    def __init__(self,training=True):
        super().__init__()
        self.resnet = resnet.resnet50(pretrained=True) # dense_feat
        self.global_context = vggnet.vggnet(input_channel=2048,opt="context")
        #self.relative_context = vggnet(input_channel=4096,opt="context")
        self.global_regressor = vggnet.vggnet(opt="regressor")
        self.training = training
        
    def forward(self, input_data_t0, input_data_t1=None):
        if self.training:
            dense_feat0 = self.resnet(input_data_t0)
            dense_feat1 = self.resnet(input_data_t1)
            #dense_feat_relative = torch.cat([dense_feat0,dense_feat1],dim=1)

            global_context_feat0 = self.global_context(dense_feat0)
            global_context_feat1 = self.global_context(dense_feat1)
            #relative_context_feat = self.relative_context(dense_feat_relative)

            global_output0,_,_ = self.global_regressor(global_context_feat0)
            global_output1,_,_ = self.global_regressor(global_context_feat1)

            return global_output0,global_output1#,relative_context_feat 
        else:
            dense_feat = self.resnet(input_data_t0)
            global_context_feat = self.global_context(dense_feat)
            global_output,_,_ = self.global_regressor(global_context_feat)
            return global_output
        