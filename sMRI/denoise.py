import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F


class DenoisingAutoencoder_legac(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2560, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(1280, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(640, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UNetDenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(UNetDenoisingAutoencoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(192, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        original_size = (x.size(2), x.size(3))
        if original_size != (128, 128):
            x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        
        conv1 = self.conv1(x)
        down1 = self.down1(conv1)
        down2 = self.down2(down1)
        conv2 = self.conv2(down2)
        
        upconv1 = self.upconv1(conv2)
        concat1 = torch.cat([upconv1, down1], dim=1)
        conv3 = self.conv3(concat1)
        
        upconv2 = self.upconv2(conv3)
        concat2 = torch.cat([upconv2, conv1], dim=1)
        
        out = self.final(concat2)
        
        if original_size != (128, 128):
            out = nn.functional.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
            
        return out


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.model = UNetDenoisingAutoencoder()
        
    def forward(self, x):
        return self.model(x)
    
def train_model(model, train_loader, test_loader, batch_size=64, epochs=30, 
                lr=0.001, device='cuda'):
    print(f"Train samples: {len(train_loader.dataset)} | Test samples: {len(test_loader.dataset)}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs('results', exist_ok=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        print(epoch)
        for batch_idx, batch in enumerate(train_loader):
            noisy_data, _ = batch
            
            noisy_data = noisy_data.to(device).float()

            optimizer.zero_grad()
            output = model(noisy_data)
            
            loss = criterion(output, noisy_data)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                noisy_data, _ = batch
                noisy_data = noisy_data.to(device).float()
                
                output = model(noisy_data)
                
                loss = criterion(output, noisy_data)
                
                val_loss += loss.item()
                val_batches += 1
                
                if batch_idx == 0:
                    n = min(8, noisy_data.size(0))
                    comparison = torch.cat([
                        noisy_data[:n],
                        output[:n]
                    ])
                    save_image_comparison(
                        comparison, 
                        f'results/epoch_{epoch+1}_comparison.png', 
                        n
                    )
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'results/denoising_autoencoder_epoch_{epoch+1}.pth')
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_curves.png')
    plt.close()
    
    torch.save(model.state_dict(), 'results/denoising_autoencoder_final.pth')
    
    return train_losses, val_losses
def save_image_comparison(comparison, filename, n=8):
    comparison = comparison.cpu().numpy()
    
    fig, axes = plt.subplots(2, n, figsize=(20, 6))
    for i in range(n):
        axes[0, i].imshow(comparison[i].reshape(224, 224), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Raw')
        
        axes[1, i].imshow(comparison[i+n].reshape(224, 224), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Denoised')

    
    plt.tight_layout()
    plt.show()




def test_denoising(model, test_dataset, batch_size=16, device='cuda'):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    
    for noisy_data, _ in test_loader:
        break
    
    noisy_data = noisy_data.to(device).float()
    
    with torch.no_grad():
        denoised_data = model(noisy_data)
    
    kernel_size = 5
    blurred_data = F.avg_pool2d(
        F.pad(noisy_data, [kernel_size//2]*4, mode='reflect'), 
        kernel_size, stride=1
    )
    
    n = min(8, noisy_data.size(0))
    
    fig, axes = plt.subplots(3, n, figsize=(20, 8))
    
    noisy_np = noisy_data.cpu().numpy()
    blurred_np = blurred_data.cpu().numpy()
    denoised_np = denoised_data.cpu().numpy()
    
    for i in range(n):
        axes[0, i].imshow(noisy_np[i].reshape(224, 224), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Noisy Input')
        
        axes[1, i].imshow(blurred_np[i].reshape(224, 224), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Simple Blur (Baseline)')
        
        axes[2, i].imshow(denoised_np[i].reshape(224, 224), cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Autoencoder Denoised')
    
    plt.tight_layout()
    plt.savefig('results/test_results.png')
    plt.close()
    
    criterion = nn.MSELoss()
    
    noisy_to_denoised_mse = criterion(noisy_data, denoised_data).item()
    
    blur_to_denoised_mse = criterion(blurred_data, denoised_data).item()
    
    noisy_to_blur_mse = criterion(noisy_data, blurred_data).item()
    
    print(f'Noisy-to-Denoised MSE: {noisy_to_denoised_mse:.6f}')
    print(f'Blur-to-Denoised MSE: {blur_to_denoised_mse:.6f}')
    print(f'Noisy-to-Blur MSE: {noisy_to_blur_mse:.6f}')
    
    if noisy_to_denoised_mse < noisy_to_blur_mse:
        print("The autoencoder preserves more original structure than simple blurring")
    else:
        print("The autoencoder is more aggressive in noise removal than simple blurring")

    print('Done!')

