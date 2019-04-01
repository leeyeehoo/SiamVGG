import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

class SiameseNet(nn.Module):

    def __init__(self, root_pretrained=None, net=None):
        super(SiameseNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),            
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1),
        )
        self.bn_adjust = nn.BatchNorm2d(1)
        self._initialize_weights()
        
        mod = models.vgg16(pretrained = True)
        
        for i in xrange(len(self.model.state_dict().items())):
            self.model.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
        
    def branch(self,allin):
        allout = self.model(allin)
        return allout

    def forward(self, z, x):
        assert z.size()[:2] == x.size()[:2]

        z = self.branch(z)
        x = self.branch(x)

        out = self.xcorr(z, x)
        out = self.bn_adjust(out)

        return out

    def xcorr(self, z, x):
        out = []
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i,:,:,:].unsqueeze(0), z[i,:,:,:].unsqueeze(0)))
        
        return torch.cat(out, dim=0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    

