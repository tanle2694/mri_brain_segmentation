import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_maxpooling=True):
        super(UnetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_maxpooling = use_maxpooling

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.out_channels,
                      out_channels=self.out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False
                      ),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class Unet(nn.Module):

    def __init__(self, in_channels, out_channels, init_features):
        super(Unet, self).__init__()
        features = init_features
        self.encoder1 = UnetBlock(in_channels, features)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UnetBlock(features, features * 2)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UnetBlock(features * 2, features * 4)
        self.pooling3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UnetBlock(features * 4, features * 8)
        self.pooling4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UnetBlock(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UnetBlock(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UnetBlock(features * 8 , features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UnetBlock(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UnetBlock(features * 2, features)

        self.conv_output = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pooling1(enc1))
        enc3 = self.encoder3(self.pooling2(enc2))
        enc4 = self.encoder4(self.pooling3(enc3))

        bottleneck = self.bottleneck(self.pooling4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        out_feature = self.conv_output(dec1)
        return torch.sigmoid(out_feature)

if __name__ == "__main__":
    net = Unet(in_channels=3, out_channels=1, init_features=32)
    net = net.to("cuda:0")
    summary(net, (3, 256, 256))