import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_bn(in_channels, out_channels, stride, conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, nlin_layer = nn.ReLU):
    return nn.Sequential(
        conv_layer(in_channels, out_channels, 3, stride, 1, bias = False),
        norm_layer(out_channels),
        nlin_layer(inplace=True)
    )


def conv_1x1(in_channels, out_channels, conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d, nlin_layer = nn.ReLU):
    return nn.Sequential(
        conv_layer(in_channels, out_channels, 1, 1, 0, bias= False),
        norm_layer(out_channels),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace = True):
        super(Hswish, self).__init__()
        self.inplace =inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace = self.inplace) / 6.0

class Hsigmoid(nn.Module):
    def __init__(self, inplace = True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x+3., inplace= self.inplace)

class SEMoudule(nn.Module):
    def __init__(self, channel, reduction = 4):
        super(SEMoudule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias = False),
            Hsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def make_divisible(x, divisible_by = 8):
    return int(np.ceil(x * 1./divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, exp, se = False, nl = 'RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1)//2
        self.use_identity = stride == 1 and in_channels == out_channels

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d

        if nl == 'RE':
            nlin_layer = nn.ReLU
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError

        if se:
            SELayer = SEMoudule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            #pw
            conv_layer(in_channels, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            #dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            #pw-linear
            conv_layer(exp, out_channels, 1, 1, 0, bias= False),
            norm_layer(out_channels),
        )

    def forward(self, x):
        if self.use_identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, num_classes = 10, input_size = 224, dropout = 0.8, mode = 'small', width_mult = 1.0):
        super(MobileNetV3, self).__init__()
        input_channels = 16
        last_channels = 1280
        if mode == 'large':
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1],
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1],
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1],
                [5, 960, 160, True, 'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'RE', 2],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1],
            ]
        else:
            raise NotImplementedError

        assert input_size % 32 == 0

        last_channels = make_divisible(last_channels * width_mult) if width_mult > 1.0 else last_channels
        self.features = [conv_bn(3, input_channels, 2, nlin_layer=Hswish)]
        self.classifier = []

        for k, exp, c, se, nl, s in mobile_setting:
            out_channels = make_divisible(c * width_mult)
            exp_channels = make_divisible(exp * width_mult)
            self.features.append(MobileBottleneck(input_channels, out_channels, k, s, exp_channels, se, nl))
            input_channels = out_channels

        #build last several layers
        if mode == 'large':
            last_conv = make_divisible(960 * width_mult)
        elif mode == 'small':
            last_conv = make_divisible(576 * width_mult)
        else:
            raise NotImplementedError

        self.features.append(conv_1x1(input_channels, last_conv, nlin_layer=Hswish))
        self.features.append(nn.AdaptiveAvgPool2d(1))
        self.features.append(nn.Conv2d(last_conv, last_channels, 1, 1, 0))
        self.features.append(Hswish(inplace=True))

        #convert to nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(last_channels, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        print(x.size())
        x = x.mean(3).mean(2) #适应不同输入尺寸
        x = self.classifier(x)
        return x



def mobilenetv3(pretrained = False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load("model.pth")
        model.load_state_dict(state_dict, strict= True)
    return model

if __name__ == '__main__':
    net = mobilenetv3()
    print(net)
    print("Total params: %.2fM \n"%(sum(p.numel() for p in net.parameters())/1000000.0))
    input_size = (1, 3, 224, 224)
    # from thop import profile
    # flops, params = profile(net, inputs = torch.randn(input_size))
    # print(flops)
    # print(params)

    x = torch.randn(input_size)
    out = net(x)
    print(out)



























