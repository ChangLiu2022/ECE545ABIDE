import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import BCEWithLogitsLoss
from model import ASDModelBinary
import torch.optim as optim
from tqdm import tqdm
class SliceDataset(Dataset):
    def __init__(self, dir_label_pairs):
        self.samples = []
        for dir_path, label in dir_label_pairs:
            files = glob.glob(os.path.join(dir_path, "*_slice*_aug*.npy"))
            self.samples.extend([(f, label) for f in files])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        img = np.load(npy_path).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        return img, torch.tensor(label, dtype=torch.float32)
    
if __name__ == '__main__':
    autism_dir = "preproc_out/autism"
    control_dir = "preproc_out/control"
    autism_label = (autism_dir, 1)
    control_label = (control_dir, 0)

    full_dataset = SliceDataset([autism_label, control_label])
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    k_folds = 5
    current_fold = 3  # Change this variable to select the fold (0 to 4)
    #0:done
    #1:done
    #2:done
    #3:todo
    #4:todo

    fold_size = len(indices) // k_folds
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k_folds)]
    
    test_idx = folds[current_fold]
    train_idx = [idx for i, fold in enumerate(folds) if i != current_fold for idx in fold]
    
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12, persistent_workers=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASDModelBinary(input_size=(1, 224, 224)).to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)
    log_train_loss = []
    log_val_loss = []
    log_train_acc = []
    log_val_acc = []
    for epoch in range(15):
        print(f"\nEpoch {epoch+1}")
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")
        log_train_acc.append(100. * correct / total)
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        print(f"Val Loss: {val_loss/len(test_loader):.4f}, Accuracy: {100. * correct / total:.2f}%")

        torch.save(model.state_dict(), f"model_paths/model_fold_{current_fold}_epoch_{epoch+1}.pth")

        log_train_loss.append(train_loss/len(train_loader))
        log_val_loss.append(val_loss/len(test_loader))
        log_val_acc.append(100. * correct / total)

    torch.save(model.state_dict(), f"model_paths/model_fold_{current_fold}_final.pth")

    np.savez(f"model_perfs/training_logs_fold_{current_fold}.npz", train_loss=log_train_loss, val_loss=log_val_loss, train_acc=log_train_acc, val_acc=log_val_acc)
    print(f"Training complete for fold {current_fold}. Model and logs saved.")
