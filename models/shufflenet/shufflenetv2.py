import torch.nn as nn
from torchsummary import summary
import torch

class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = (1, 1), padding = 0):
        super(Conv2dBnRelu, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,out_channels, kernel_size,stride, padding= padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseBn(nn.Module):
    def __init__(self, in_channels, kernel_size, stride =(1, 1), padding = 0):
        super(DepthwiseBn, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return x


def channelShuffle(inputs, groups):
    in_shape = inputs.size()
    b, in_channels, h, w = in_shape
    assert in_channels % groups == 0
    x = inputs.view((b, in_channels//groups, groups, h, w))
    x = torch.transpose(x,1, 2).contiguous()
    x = x.view((b, in_channels, h, w))
    # print(in_shape)
    return x


class ShufflenetUnit(nn.Module):
    def __init__(self, in_channels):
        super(ShufflenetUnit, self).__init__()
        assert in_channels%2 == 0
        self.in_channels = in_channels
        self.conv2d_bn_relu = Conv2dBnRelu(in_channels//2, in_channels//2, (1, 1), (1, 1))
        self.dconv_bn = DepthwiseBn(in_channels//2, (3, 3), (1, 1), 1)

    def forward(self, x):
        shortcut, x = torch.split(x, x.size()[1]//2, dim = 1)
        x = self.conv2d_bn_relu(x)
        x = self.dconv_bn(x)
        x = self.conv2d_bn_relu(x)
        x = torch.concat((shortcut, x), dim = 1)
        return x


class ShufflenetUnit2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShufflenetUnit2, self).__init__()
        assert out_channels%2 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1_bn_relu = Conv2dBnRelu(in_channels, out_channels//2, (1, 1), (1, 1))
        self.dconv_bn = DepthwiseBn(out_channels//2, (3, 3), (2, 2), 1)
        self.conv2_bn_relu = Conv2dBnRelu(out_channels//2, out_channels - in_channels, (1,1), (1,1))

        self.shortcut_dconv_bn = DepthwiseBn(in_channels, (3,3), (2,2), 1)
        self.shortcut_conv_bn_relu = Conv2dBnRelu(in_channels, in_channels, (1,1),(1,1))

    def forward(self, input):
        shortcut, x = input, input
        x = self.conv1_bn_relu(x)
        x = self.dconv_bn(x)
        x = self.conv2_bn_relu(x)
        shortcut = self.shortcut_dconv_bn(shortcut)
        shortcut = self.shortcut_conv_bn_relu(shortcut)
        x = torch.cat([x, shortcut], dim=1)
        x = channelShuffle(x, 2)
        return x


class ShufflenetStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ShufflenetStage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ops = []
        for i in range(num_blocks):
            if i == 0:
                op = ShufflenetUnit2(in_channels, out_channels)
            else:
                op = ShufflenetUnit(out_channels)
            self.ops.append(op)

    def forward(self, inputs):
        x = inputs
        for op in self.ops:
            x = op(x)
        return x





class shuffleNetV2(nn.Module):
    def __init__(self, input_channel, num_classes, channels_per_stage=(116, 232, 464)):
        super(shuffleNetV2, self).__init__()
        self.num_classes = num_classes
        self.conv1_bn_relu = Conv2dBnRelu(3, input_channel, 3, 2, 1)
        self.pool1 = nn.MaxPool2d(3,2) #padding: same
        self.stage2 = ShufflenetStage(input_channel, channels_per_stage[0], 4)
        self.stage3 = ShufflenetStage(channels_per_stage[0], channels_per_stage[1], 8)
        self.stage4 = ShufflenetStage(channels_per_stage[1], channels_per_stage[2], 4)
        self.conv5_bn_relu = Conv2dBnRelu(channels_per_stage[2], 1024, 1, 1)
        self.linear = nn.Linear(1024, 1000)


    def forward(self, inputs):
        x = self.conv1_bn_relu(inputs)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5_bn_relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size()[:2])
        x = self.linear(x)
        return x

if __name__ == '__main__':
    net = shuffleNetV2(24,1000)
    print(net)
    print("Total params: %.2fM \n"%(sum(p.numel() for p in net.parameters())/1000000.0))
    input_size = (1, 3, 224, 224)
    # from thop import profile
    # flops, params = profile(net, inputs = torch.randn(input_size))
    # print(flops)
    # print(params)
    print(summary(net, ( 3, 224, 224)))
    x = torch.randn(input_size)
    out = net(x)
    # print(out)







