import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential (
            nn.Conv1d(in_channels=1, out_channels=20, kernel_size=(1,4)),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,4)),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,4)),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=(1,1)),
            nn.ELU(),
            #nn.PixelShuffle(4),
            )
        self.fc = nn.Linear(1856, 6)

    def forward(self, x):
        x=self.conv1(x)
        x=x.view(x.size(0),-1)
        x = self.fc(x)
        return x

class AECNN(nn.Module):
    def __init__(self):
        super(AECNN, self).__init__()
        self.encoder = nn.Sequential (
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=(1,4)),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,4)),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((1, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,4)),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=(1,6), stride=(1,2)),
            nn.ELU(),
            #nn.UpsamplingBilinear2d(scale_factor=(1,2)),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=(1,5), stride=(1,2)),
            nn.ELU(),
            #nn.UpsamplingBilinear2d(scale_factor=(1,2)),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=(1,4), stride=(1,2)),
            nn.ELU(),
            #nn.PixelShuffle(4),
            )
        self.fc = nn.Linear(1856, 6)

    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

#Encoder
class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        self.conv1=nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=(1,4)), nn.ELU())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=(1,4)), nn.ELU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 4)), nn.ELU(),
                                   nn.BatchNorm2d(32))
        self.maxpool= nn.MaxPool2d((1, 2))
    def forward(self, x):
        x = self.conv1(x)
        x= self.maxpool(x)
        x_out = self.conv2(x)
        xgauss = self.maxpool(x_out)
        x_var = self.conv3(x)
        x_vargauss = self.maxpool(x_var)
        return xgauss, x_vargauss

# Decoder
class P_net(nn.Module):
    def __init__(self):
        super(P_net, self).__init__()
        self.conv1 = nn.Sequential(nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=(1,6), stride=(1,2)),
            nn.ELU())
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=(1, 6), stride=(1, 2)),
            nn.ELU())

    def forward(self, x):
        x = self.conv1(x)
        xgauss = self.conv2(x)
        return xgauss

# Discriminator
class D_net_gauss(nn.Module):
    def __init__(self):
        super(D_net_gauss, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=(1, 6), stride=(1, 2)),
            nn.ELU())
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=(1, 6), stride=(1, 2)),
            nn.ELU())
        self.fc= nn.Linear(256,1)
    def forward(self, x):
        x = self.conv1(x)
        x= self.conv2(x)
        return F.sigmoid(self.fc(x))