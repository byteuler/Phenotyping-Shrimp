import torch
import torch.nn as nn
from torchvision import models
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
preresnet = models.resnet50(pretrained=True)
class GCNconv(nn.Module):
    def __init__(self,in_channels,num_classes, k=15):
        """
        :param in_channels:
        :param num_classes:
        :param k:
        """
        super(GCNconv, self).__init__()
        pad = (k-1 ) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=num_classes,kernel_size=(k,1),padding=(pad,0),bias=False),
            nn.Conv2d(in_channels=num_classes,out_channels=num_classes,kernel_size=(1,k),padding=(0,pad),bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=num_classes,kernel_size=(1,k),padding=(0,pad),bias=False),
            nn.Conv2d(in_channels=num_classes,out_channels=num_classes,kernel_size=(k,1),padding=(pad,0),bias=False)
        )
    def forward(self,input):
        x1 = self.conv1(input)
        x2 = self.conv2(input)

        assert x1.shape == x2.shape

        return x1 + x2

class BR(nn.Module):
    def __init__(self,in_channels):
        """
        :param channel: in_channel eq num_classes
        """
        super(BR, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3, padding =1, bias =False),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1,bias=False),
        )
    def forward(self,input):
        x = self.shortcut(input)
        return x + input
class encoder_decoder(nn.Module):
    def __init__(self,in_channels,num_classes,k = 15):
        """
        :param in_channels:
        :param num_classes: out_channel eq num_classes
        :param k:
        """
        super(encoder_decoder, self).__init__()
        self.gcn = GCNconv(in_channels,num_classes,k)
        self.br = BR(num_classes)
        self.deconv = nn.ConvTranspose2d(in_channels=num_classes,out_channels=num_classes,kernel_size=4,stride=2,padding=1,bias=False) 

    def forward(self, x1,x2=None):
        x1 = self.gcn(x1)
        x1 = self.br(x1)
        if x2 ==None:
            x = self.deconv(x1)
        else:
            x = x1 + x2
            x = self.br(x)
            x = self.deconv(x)
        return x
class GCN(nn.Module):
    def __init__(self,numclass, k = 15):
        super(GCN, self).__init__()
        self.num_class = numclass
        self.k = k
        self.layer0 = nn.Sequential(
            preresnet.conv1,
            preresnet.bn1,
            preresnet.relu
        )
        self.layer1 = nn.Sequential(
            preresnet.maxpool,
            preresnet.layer1
        )
        self.layer2 = preresnet.layer2
        self.layer3 = preresnet.layer3
        self.layer4 = preresnet.layer4
        self.br = BR(self.num_class)
        self.deconv = nn.ConvTranspose2d(self.num_class,self.num_class,4,2,1,bias=False)
        self.branch4 = encoder_decoder(2048,self.num_class,self.k)
        self.branch3 = encoder_decoder(1024,self.num_class,self.k)
        self.branch2 = encoder_decoder(512,self.num_class,self.k)
        self.branch1 = encoder_decoder(256,self.num_class,self.k)
    def forward(self,input):
        x0 = self.layer0(input)  
        x1 = self.layer1(x0)  
        x2 = self.layer2(x1)  
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)


        x4 = self.branch4(x4)
        x3 = self.branch3(x3,x4)
        x2 = self.branch2(x2,x3)
        x1 = self.branch1(x1,x2)
        x = self.br(x1)
        x = self.deconv(x)
        x = self.br(x)

        return x

if __name__ == "__main__":

    rgb = torch.randn(1,3,480,1600)
    model = GCN(7,k=15)
    model.to(device)
    rgb = rgb.to(device)
    out = model(rgb)
    print(out.shape)