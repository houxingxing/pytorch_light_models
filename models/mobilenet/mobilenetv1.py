
import torch.nn as nn
from torchsummary import summary

class MobileNetV1(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(MobileNetV1, self).__init__()
        self.model = nn.Sequential(
            self.conv_bn(input_channel, 32, 2),
            self.conv_dw(32, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            #repeat 5 times
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            #
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024, 1024, 2),
            nn.AdaptiveAvgPool2d(1),

        )
        self.fc = nn.Linear(1024, num_classes)

    def conv_bn(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def conv_dw(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, padding = 1, groups = in_channels, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True),

            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    #model check
    model = MobileNetV1(3, 1000)
    summary(model, input_size = (3, 224, 224), device = 'cpu')
