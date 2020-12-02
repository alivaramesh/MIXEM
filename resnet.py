'''
Adapted from https://github.com/sthalles/SimCLR/blob/master/models/resnet_simclr.py
'''

import sys
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet(nn.Module):

    def __init__(self, base_model, out_dim,conv1_k3=False,**kwargs):
        
        super(ResNet, self).__init__()

        self.mixture = 'Mixture' in kwargs
        if self.mixture:
            self.n_comps = kwargs['Mixture']['n_comps']
        
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}
        print('Resnet conv1_k3:',conv1_k3)
        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features
        
        if conv1_k3:
            self.conv1    = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1      = nn.BatchNorm2d(64)
            ops = [self.conv1, self.bn1, nn.ReLU()]  + list(resnet.children())[4:-1]
        elif kwargs.get('maxpool',False):
            maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
            ops = list(resnet.children())[:-2] + [maxpool]
        else:
            ops = list(resnet.children())[:-1]
        
        self.features = nn.Sequential(*ops)
        
        inf= num_ftrs
        assert inf>= out_dim

        if self.mixture:
            self.mixtures = nn.ModuleList([nn.Sequential(nn.Linear(inf, inf),nn.ReLU(),nn.Linear(inf, out_dim)) for i in range(self.n_comps)])
            self.pi = nn.Sequential(nn.Linear(inf, inf),nn.ReLU(),nn.Linear(inf, self.n_comps))
        else:
            self.l1 = nn.Linear(inf, inf)
            self.l2 = nn.Linear(inf, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Base feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        
        h = self.features(x)
        h = h.squeeze()
        hin = h

        if self.mixture:
            pi = self.pi(hin)
            mixtures = torch.stack([self.mixtures[i](hin) for i in range(self.n_comps)],dim=1)
            return h, (pi,mixtures)
        else:
            x = self.l1(hin)
            x = F.relu(x)
            x = self.l2(x)
        return h, x