import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(in_channels=16, out_channels=16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(in_channels=16, out_channels=32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(in_channels=32, out_channels=64, num_blocks=3, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(64, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += shortcut
        x = self.relu(x)
        return x

# Instantiate the model
resnet20 = ResNet20()
print(resnet20)

#class BasicBlock(nn.Module):
#    expansion = 1

#    def __init__(self, in_channels, out_channels, stride=1):
#        super(BasicBlock, self).__init__()
#        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#        self.batch_norm1 = nn.BatchNorm2d(out_channels)
#        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#        self.batch_norm2 = nn.BatchNorm2d(out_channels)

 #       self.shortcut = nn.Sequential()
 #       if stride != 1 or in_channels != self.expansion * out_channels:
 #           self.shortcut = nn.Sequential(
 #               nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
 #               nn.BatchNorm2d(self.expansion * out_channels)
 #           )

 #   def forward(self, x):
 #       residual = x
 #       out = F.relu(self.batch_norm1(self.conv1(x)))
 #       out = self.batch_norm2(self.conv2(out))
 #       out += self.shortcut(residual)
 #       out = F.relu(out)
 #       return out

#class ResNet(nn.Module):
#    def __init__(self, block, num_blocks, num_classes=10):
#        super(ResNet, self).__init__()
#        self.in_channels = 16

#        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#        self.batch_norm = nn.BatchNorm2d(16)
#        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#        self.fc = nn.Linear(64, num_classes)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight, 1)
#                nn.init.constant_(m.bias, 0)

#    def _make_layer(self, block, channels, num_blocks, stride):
#        strides = [stride] + [1] * (num_blocks - 1)
#        layers = []
#        for stride in strides:
#            layers.append(block(self.in_channels, channels, stride))
#            self.in_channels = channels * block.expansion
#        return nn.Sequential(*layers)

#    def forward(self, x):
#        out = F.relu(self.batch_norm(self.conv1(x)))
#        out = self.layer1(out)
#        out = self.layer2(out)
#        out = self.layer3(out)
#        out = self.avg_pool(out)
#        out = out.view(out.size(0), -1)
#        out = self.fc(out)
#        return out

#def ResNet20():
#    return ResNet(BasicBlock, [3, 3, 3])

# Convolution block with BatchNormalization
#def ConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
#    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
#              nn.BatchNorm2d(out_channels),
#              nn.ReLU(inplace=True)]
#    return nn.Sequential(*layers)

# Residual block
#class ResidualBlock(nn.Module):
#    def __init__(self, in_channels, out_channels, stride=1):
#        super(ResidualBlock, self).__init__()
#        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
#        self.conv2 = ConvBlock(out_channels, out_channels)

        # Adjusting dimensions for residual connection if needed
#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_channels != out_channels:
#            self.shortcut = nn.Sequential(
#                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(out_channels)
#            )

#    def forward(self, x):
#        out = self.conv1(x)
#        out = self.conv2(out)
#        out += self.shortcut(x)
#        out = nn.ReLU(inplace=True)(out)
#        return out

# ResNet20 architecture
#class ResNet20(nn.Module):
#    def __init__(self, num_classes=1000):
#        super(ResNet20, self).__init__()
#        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
#        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#        self.conv2_x = self.make_residual_layers(64, 64, num_blocks=2)
#        self.conv3_x = self.make_residual_layers(64, 128, num_blocks=2, stride=2)
 #       self.conv4_x = self.make_residual_layers(128, 256, num_blocks=2, stride=2)
# #       self.conv5_x = self.make_residual_layers(256, 512, num_blocks=2, stride=2)
# #       self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
  #      self.fc = nn.Linear(512, num_classes)

   # def forward(self, x):
    #    out = self.conv1(x)
     #   out = self.maxpool(out)
      #  out = self.conv2_x(out)
       # out = self.conv3_x(out)
       # out = self.conv4_x(out)
       # out = self.conv5_x(out)
       # out = self.avgpool(out)
       # out = torch.flatten(out, 1)
       # out = self.fc(out)
       # return out

#    def make_residual_layers(self, in_channels, out_channels, num_blocks, stride=1):
#        layers = []
#        layers.append(ResidualBlock(in_channels, out_channels, stride))
#        for _ in range(1, num_blocks):
#            layers.append(ResidualBlock(out_channels, out_channels))
#        return nn.Sequential(*layers)

# Create an instance of ResNet20
#model = ResNet20(num_classes=1000)  # Example with 1000 output classes
#print(model)
