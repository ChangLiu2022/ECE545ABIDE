import torch
import torch.nn as nn
import torch.nn.functional as F

class ASDModelBinary(nn.Module):
    def __init__(self, input_size=(1, 224, 224)):
        super(ASDModelBinary, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=1, padding=0)
        
        self.conv2 = nn.Conv2d(48, 24, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        
        self.conv3 = nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(24)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=1, padding=0)
        
        self.conv4 = nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(24)
        self.pool4 = nn.MaxPool2d(kernel_size=5, stride=1, padding=0)
        
        self.conv5 = nn.Conv2d(24, 16, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(16)
        self.pool6 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)
        
        self.conv7 = nn.Conv2d(16, 16, kernel_size=5, stride=2, padding=0)
        self.bn7 = nn.BatchNorm2d(16)
        self.pool7 = nn.MaxPool2d(kernel_size=4, stride=1, padding=1)
        
        self.dropout = nn.Dropout(0.6)
        dummy = torch.zeros(1, *input_size)
        out = self._features(dummy)
        flattened_dim = out.view(-1).size(0)
        
        self.fc1 = nn.Linear(flattened_dim, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def _features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(x))
        x = F.relu(self.pool1(x))
        
        x = F.relu(self.conv2(x))
        x = F.relu(self.bn2(x))
        x = F.relu(self.pool2(x))
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn3(x))
        x = F.relu(self.pool3(x))
        
        x = F.relu(self.conv4(x))
        x = F.relu(self.bn4(x))
        x = F.relu(self.pool4(x))
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.bn5(x))
        x = F.relu(self.pool5(x))
        
        x = F.relu(self.conv6(x))
        x = F.relu(self.bn6(x))
        x = F.relu(self.pool6(x))
        
        x = F.relu(self.conv7(x))
        x = F.relu(self.bn7(x))
        x = F.relu(self.pool7(x))
        
        return x

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x).squeeze(1)  # no sigmoid
        return x


class OptimizedASDModel(nn.Module):
    def __init__(self, input_size=(1, 224, 224)):
        super(OptimizedASDModel, self).__init__()
        
        # Reduced number of layers and simplified architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        dummy = torch.zeros(1, *input_size)
        x = self._features(dummy)
        flattened_dim = x.view(-1).size(0)
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(flattened_dim, 128)
        self.fc2 = nn.Linear(128, 1)
    
    def _features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        return x

    def forward(self, x):
        x = self._features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x).squeeze(1)  # no sigmoid
        return x