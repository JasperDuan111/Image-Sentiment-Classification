import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super().__init__()
        
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )
        
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.BatchNorm2d(reduce_3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )

        # 使用两层3x3卷积来代替5x5卷积
        # 这样可以减少参数量和计算量，同时保持感受野
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.BatchNorm2d(reduce_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5, out_5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )
        
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class InceptionNet(nn.Module):
    def __init__(self): 
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), 
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 第一次下采样：48x48 -> 24x24
        )
        
        # 输出通道数计算方法： out_1x1 + out_3x3 + out_5x5 + pool_proj
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)  # 输出: 64+128+32+32 = 256
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)  # 输出: 128+192+96+64 = 480 

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 第二次下采样：24x24 -> 12x12

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)  # 输出: 192+208+48+64 = 512
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)  # 输出: 160+224+64+64 = 512
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)  # 输出: 128+256+64+64 = 512
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)  # 输出: 112+288+64+64 = 528
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)  # 输出: 256+320+128+128 = 832 

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 第三次下采样：12x12 -> 6x6

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)  # 输出: 256+320+128+128 = 832
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)  # 输出: 384+384+128+128 = 1024 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 7)  
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x) 
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x
    
class InceptionNet_simplified(nn.Module):
    def __init__(self): 
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), 
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 第一次下采样：48x48 -> 24x24
        )
        
        # 输出通道数计算方法： out_1x1 + out_3x3 + out_5x5 + pool_proj
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)  # 输出: 64+128+32+32 = 256
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)  # 输出: 128+192+96+64 = 480 

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 第二次下采样：24x24 -> 12x12

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)  # 输出: 192+208+48+64 = 512
        self.inception4b = InceptionBlock(512, 112, 144, 288, 32, 64, 64)  # 输出: 112+288+64+64 = 528

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 第三次下采样：12x12 -> 6x6

        self.inception5a = InceptionBlock(528, 256, 160, 320, 32, 128, 128)  # 输出: 256+320+128+128 = 832

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(832),
            nn.Dropout(0.3),
            nn.Linear(832, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

class Mynet_v1(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_avgpool(x)
        x = self.fc(x)
        return x

class Mynet_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception = InceptionBlock(128, 48, 48, 96, 24, 48, 24)  # 输出通道数: 48+96+48+24=216

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(216, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.inception(x)
        x = self.global_avgpool(x)
        x = self.fc(x)
        return x