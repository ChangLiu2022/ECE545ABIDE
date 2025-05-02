import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import BCEWithLogitsLoss
from model import ASDModelBinary
from model import ASDModelBinary_Small
import torch.optim as optim
from tqdm import tqdm
class SliceDataset(Dataset):
    def __init__(self, dir_label_pairs):
        self.samples = []
        for dir_path, label in dir_label_pairs:
            files = glob.glob(os.path.join(dir_path, "*_slice*_aug0.npy"))
            self.samples.extend([(f, label) for f in files])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        img = np.load(npy_path).astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        return img, torch.tensor(label, dtype=torch.float32)
    
if __name__ == '__main__':
    k_folds = [0, 1, 2, 3, 4]
    for current_fold in k_folds:
        torch.cuda.empty_cache()
        base_dir = "preproc_out"

        groups = [f"group{i+1}" for i in range(5)]
        test_group = groups[current_fold]
        train_groups = [group for group in groups if group != test_group]

        train_dirs = [(os.path.join(base_dir, group, "autism"), 1) for group in train_groups] + \
                    [(os.path.join(base_dir, group, "control"), 0) for group in train_groups]
        test_dirs = [(os.path.join(base_dir, test_group, "autism"), 1), 
                    (os.path.join(base_dir, test_group, "control"), 0)]

        train_dataset = SliceDataset(train_dirs)
        test_dataset = SliceDataset(test_dirs)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12, persistent_workers=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12, persistent_workers=True, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ASDModelBinary_Small().to(device)
        criterion = BCEWithLogitsLoss()
        optimizer = optim.SGD(
                                model.parameters(),
                                lr=3e-3, #was 1e-4
                                momentum=0.9,
                                weight_decay=3e-4 #was 1e-3
                            )
        log_train_loss = []
        log_val_loss = []
        log_train_acc = []
        log_val_acc = []
        for epoch in range(10):
            print(f"\nEpoch {epoch+1}")
            model.train()
            train_loss, correct, total = 0.0, 0, 0
            for images, labels in tqdm(train_loader, desc=f"Fold {current_fold} Training"):
                images, labels = images.to(device, non_blocking=True), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(torch.sigmoid(outputs), labels)
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
                    val_loss += criterion(torch.sigmoid(outputs), labels).item()
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
