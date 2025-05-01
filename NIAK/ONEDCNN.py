import torch
import torch.nn as nn
import torch.nn.functional as F

class ONEDCNN(nn.Module):
    def __init__(self, n_features = 19900, channels =1 , n_outputs= 2, std = 0.1):
        super(ONEDCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=64, kernel_size=5)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.std = std
        # Calculate flattened dimension after conv and pooling
        self.flattened_dim = 64 * ((n_features - 4) // 2)
        
        self.dense_layers = nn.Sequential(
            nn.Linear(self.flattened_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.output = nn.Linear(8, n_outputs)
        
        # L1 regularization strength for the 32-unit layer
        self.l1_strength = 0.0026

    def forward(self, x):
        # Adjust input shape for PyTorch Conv1d (channels first)
        #x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = F.elu(self.conv1(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        if self.training:
           noise = torch.randn_like(x) * self.std
           x = x*(1 + noise)
        x = self.dense_layers(x)
        return F.softmax(self.output(x), dim=1)

# Note: During training, remember to add L1 regularization manually:
# loss = criterion(outputs, targets)
# l1_loss = model.l1_strength * sum(p.abs().sum() for p in model.dense_layers[4].parameters())
# total_loss = loss + l1_loss