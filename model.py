import torch
import torch.nn as nn
import torch.nn.functional as F


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_planes,planes,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,
                               kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,
                               kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        self.shortcut = nn.Sequential()
        if stride !=1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self,x):
        # add = self.shortcut(x)
        # print('add:',add.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # print('out:', out.shape)
        # out = out + add
        out = F.relu(out)


        return out

# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,
                               kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes,planes,
                               kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes,self.expansion*planes,
                               kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,
                          stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

def Conv1(in_planes, planes):
    return nn.Sequential(
        nn.Conv2d(in_planes,planes,7,2,3,bias=False),
        nn.BatchNorm2d(planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3,2,1)
    )

class ResNet(nn.Module):
    def __init__(self,block,num_block,num_classes=100):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = Conv1(3,64)

        self.layer1 = self._make_layer(block,64,num_block[0],stride=1)
        self.layer2 = self._make_layer(block,128,num_block[1],stride=2)
        self.layer3 = self._make_layer(block,256,num_block[2],stride=2)
        self.layer4 = self._make_layer(block,512,num_block[3],stride=2)

        self.linear = nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self,block,planes,num_blocks,stride):
        strides = [stride] + [1] * (num_blocks -1)
        # print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes = planes * block.expansion
            # print(self.in_planes)

        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out,4)
        out = out.view(out.size(0),-1)
        out = self.linear(out)

        return out


def ResNet18():
    return ResNet(BasicBlock,[2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock,[3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck,[3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck,[3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck,[3,8,36,3])

def test():
    net = ResNet152()
    y = net(torch.randn(1,3,224,224))
    print(y.size())
