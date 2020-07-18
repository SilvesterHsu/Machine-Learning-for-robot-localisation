import torch
import torch.nn as nn
#from .utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url

class Context(nn.Module):
    def __init__(self, input_channel=2048):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=128,kernel_size=1),
            nn.ReLU()
        )
        self.context5_1 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.context5_2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.context5_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.context5_4 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.squeeze2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.squeeze(x)
        x = self.context5_1(x) #context5_1 
        x = self.context5_2(x) #context5_2
        x = self.context5_3(x) #context5_3
        x = self.context5_4(x) #context5_4
        x = self.squeeze2(x)
        return x
    
class Regressor(nn.Module):
    def __init__(self, in_features=6400):
        super().__init__()
        self.flatten = nn.Flatten()
        # Part 1: trans
        self.fc1_trans = nn.Sequential(
            nn.Linear(in_features=in_features,out_features=4096),
            nn.ReLU()
        )
        self.fc2_trans = nn.Sequential(
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU()
        )
        self.fc3_trans = nn.Sequential(
            nn.Linear(in_features=4096,out_features=128),
            nn.ReLU()
        )
        self.logits_t = nn.Linear(in_features=128,out_features=3)
        
        # Part 2: rot
        self.fc1_rot = nn.Sequential(
            nn.Linear(in_features=in_features,out_features=4096),
            nn.ReLU()
        )
        self.fc2_rot = nn.Sequential(
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU()
        )
        self.fc3_rot = nn.Sequential(
            nn.Linear(in_features=4096,out_features=128),
            nn.ReLU()
        )
        self.logits_r = nn.Linear(in_features=128,out_features=4)
        
    def forward(self, x):
        x = self.flatten(x)
        # Part 1: trans
        net_t = self.fc1_trans(x)
        net_t = self.fc2_trans(net_t)
        feature_t = self.fc3_trans(net_t)        
        # Part 2: rot
        net_r = self.fc1_rot(x)
        net_r = self.fc2_rot(net_r)
        feature_r = self.fc3_rot(net_r)
        # Part 3: FC layer
        logits_t = self.logits_t(feature_t)
        logits_r = self.logits_r(feature_r)
        
        logits_r = nn.functional.normalize(logits_r, p=2, dim=1)

        logits = torch.cat([logits_t, logits_r],dim=1)
        return logits, feature_t, feature_r
    
class vggnet(nn.Module):
    def __init__(self, opt="context", input_channel = 2048):
        super().__init__()
        self.opt = opt
        if opt == "context":
            self.context = Context(input_channel)
        elif opt == "regressor":
            self.regressor = Regressor()
        
    def forward(self,x):
        if self.opt == "context":
            x = self.context(x)
            return x
        elif self.opt == "regressor":
            logits,feat_t,feat_r = self.regressor(x)
            return logits,feat_t,feat_r