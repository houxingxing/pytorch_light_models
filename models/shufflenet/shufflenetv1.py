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

class GroupConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, stride):
        super(GroupConvBn, self).__init__()
        self.gconv = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, groups = groups, stride = stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.gconv(x)
        x = self.bn(x)
        x = self.relu(x)
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


class ShufflenetUnitA(nn.Module):
    def __init__(self, in_channels, out_channels, groups = 3):
        super(ShufflenetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels%4 == 0
        self.in_channels = in_channels
        self.groups = groups
        self.bottleneck_channels = out_channels //4
        self.gconv1_bn_relu = GroupConvBn(in_channels, self.bottleneck_channels, (1, 1), self.groups, (1,1))
        self.dconv_bn = DepthwiseBn(self.bottleneck_channels, (3, 3), (1, 1), 1)
        self.gconv2_bn_relu = GroupConvBn(self.bottleneck_channels, self.in_channels, (1, 1), self.groups, (1, 1))
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, input):
        x = input
        x = self.gconv1_bn_relu(x)
        x = channelShuffle(x, self.groups)
        x = self.dconv_bn(x)
        x = self.gconv2_bn_relu(x)
        x = x + input
        x = self.relu1(x)
        return x



class ShufflenetUnitB(nn.Module):
    def __init__(self, in_channels, out_channels, groups = 3):
        super(ShufflenetUnitB, self).__init__()
        assert out_channels%4 == 0
        self.in_channels = in_channels
        self.out_channels = out_channels - in_channels
        self.groups = groups
        self.bottleneck_channels = out_channels //4
        self.gconv1_bn_relu = GroupConvBn(in_channels, self.bottleneck_channels, (1, 1), self.groups, (1,1))
        self.dconv_bn = DepthwiseBn(self.bottleneck_channels, (3, 3), (2, 2), 1)
        self.gconv2_bn_relu = GroupConvBn(self.bottleneck_channels, self.out_channels, (1, 1), self.groups, (1, 1))
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, input):
        x = input
        x = self.gconv1_bn_relu(x)
        x = channelShuffle(x, self.groups)
        x = self.dconv_bn(x)
        x = self.gconv2_bn_relu(x)
        input = nn.functional.max_pool2d(input, 3, stride = 2, padding = 1)
        x = torch.cat([input, x], dim = 1)
        x = self.relu1(x)
        return x



class ShufflenetStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, groups):
        super(ShufflenetStage, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ops = []
        for i in range(num_blocks):
            if i == 0:
                op = ShufflenetUnitB(in_channels, out_channels, groups)
            else:
                op = ShufflenetUnitA(out_channels, out_channels, groups)
            self.ops.append(op)

    def forward(self, inputs):
        x = inputs
        for op in self.ops:
            x = op(x)
        return x


class ShuffleNetV1(nn.Module):
    def __init__(self, in_channels, num_classes, channels_per_stage = (24, 240, 480, 960), groups = 3):
        super(ShuffleNetV1, self).__init__()
        self.conv1_bn_relu = Conv2dBnRelu(in_channels, channels_per_stage[0], (3,3), (2,2), 1)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.stage2 = ShufflenetStage(channels_per_stage[0], channels_per_stage[1], 4, groups)
        self.stage3 = ShufflenetStage(channels_per_stage[1], channels_per_stage[2], 8, groups)
        self.stage4 = ShufflenetStage(channels_per_stage[2], channels_per_stage[3], 4, groups)
        self.fc5 = nn.Linear(channels_per_stage[3], num_classes)

    def forward(self, x):
        x = self.conv1_bn_relu(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = nn.functional.max_pool2d(x, x.size()[2:])
        x = x.view(x.size()[0], -1)
        x = self.fc5(x)
        logits = nn.functional.softmax(x)
        return logits

if __name__== "__main__":
    net = ShuffleNetV1(3, 1000)
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