import torch
from torchsummary import summary
from torch import nn

class Conv2d_Bn_Relu(nn.Module):
    def __init__(self, out_channels):
        super(Conv2d_Bn_Relu, self).__init__()
        self.conv2d = nn.Conv2d(3, out_channels= out_channels,kernel_size=(3, 3), stride = (2,2), padding=1, bias = False) #half size
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = input
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, padding = 0):
        super(Conv_1x1, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv2d(input)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding = 1):
        super(DepthwiseConv, self).__init__()
        self.depthconv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.Hardswish(inplace=True)

        # self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        # self.bn2 = nn.BatchNorm2d(out_channels),
        # self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.depthconv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class Swish(nn.Module):
    def __init__(self, inplace):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)




class SEModule(nn.Module):
    def __init__(self, in_channels, reduction):
        super(SEModule, self).__init__()
        # self.avg = nn.AvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels//reduction, bias=False)
        self.swish1 = Swish(inplace=True)
        self.fc2 = nn.Linear(in_channels//reduction, in_channels, bias=False)
        self.swish2 = Swish(inplace=True)


    def forward(self, input):
        x = nn.AvgPool2d(input.size()[2:])(input)
        x = x.view(x.size()[0], x.size()[1])
        x = self.fc1(x)
        x = self.swish1(x)
        x = self.fc2(x)
        x = self.swish2(x).view(x.size()[0], x.size()[1], 1, 1)
        out = input * x.expand_as(input)
        return out




class DropPath(nn.Module):
    def __init__(self, drop_prob = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_connection(self, x, drop_prba = 0.0):
        keep_proba = 1 - drop_prba
        shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
        # print("shape: ", shape)
        random_tensor = keep_proba + torch.rand(shape, dtype = x.dtype, device = x.device)
        random_tensor.floor_()
        # print("random_tensor", random_tensor, random_tensor.size())
        output = x.div(keep_proba) * random_tensor
        return output

    def forward(self, input):
        return self.drop_connection(input, self.drop_prob)




class MBConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, stride, padding, reduction):
        super(MBConv, self).__init__()
        self.conv1 = Conv_1x1(in_channels, mid_channels, padding=padding)
        self.dw_conv2 = DepthwiseConv(mid_channels, kernel_size, stride)
        self.se3 = SEModule(mid_channels, reduction)
        self.conv4 = Conv_1x1(mid_channels, out_channels)
        self.skip_connection = True if in_channels == out_channels else False
        self.drop_path = DropPath(0.5)

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.dw_conv2(x)
        x = self.se3(x)
        x = self.conv4(x)
        if self.skip_connection:
            x = self.drop_path(x)
            x = torch.add(input, x)
            return x
        else:
            return x

class EfficientNetB0(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EfficientNetB0, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channels)
        self.mbconv2 = MBConv(in_channels, 64, 16, (3,3), (1, 1), 0, 4)
        self.mbconv3 = MBConv(16, 32, 24, (3,3), (1, 1), 0, 4)
        self.mbconv31 = MBConv(24, 32, 24, (3, 3), (1, 1), 0, 4)
        self.mbconv4 = MBConv(24, 32, 40, (5, 5), (2, 2), 1, 4)
        self.mbconv41 = MBConv(40, 32, 40, (5, 5), (1, 1), 1, 4)
        self.mbconv5 = MBConv(40, 32, 80, (5, 5), (2, 2), 1, 4)
        self.mbconv51 = MBConv(80, 32, 80, (5, 5), (1, 1), 1, 4)
        self.mbconv6 = MBConv(80, 32, 112, (5, 5), (2, 2), 1, 4)
        self.mbconv61 = MBConv(112, 32, 112, (5, 5), (1, 1), 1, 4)
        self.mbconv7 = MBConv(112, 32, 192, (5, 5), (1, 1), 1, 4)
        self.mbconv71 = MBConv(192, 32, 192, (5, 5), (1, 1), 1, 4)
        self.mbconv8 = MBConv(192, 32, 320, (3, 3), (2, 2), 0, 4)
        self.conv9 = Conv_1x1(320, 1290)
        self.fc10 = nn.Linear(1290, num_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        for i in range(2):
            x = self.mbconv31(x)
        x = self.mbconv4(x)
        for i in range(2):
            x = self.mbconv41(x)
        x = self.mbconv5(x)
        for i in range(3):
            x = self.mbconv51(x)
        x = self.mbconv6(x)
        for i in range(3):
            x = self.mbconv61(x)
        x = self.mbconv7(x)
        for i in range(4):
            x = self.mbconv71(x)
        x = self.mbconv8(x)
        x = self.conv9(x)
        x = nn.AvgPool2d(x.size()[2:])(x)
        x = x.view(x.size()[:2])
        x = self.fc10(x)
        return x

if __name__ == "__main__":
    #model check
    model = EfficientNetB0(32, 1000)
    summary(model, input_size = (3, 224, 224), device = 'cpu')