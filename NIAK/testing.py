import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CSVloader import RowWiseCSVLoader

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            #labels = labels.to(device).float().view(-1, 1)  # Ensure shape compatibility

            outputs = model(inputs)
            
            labels = labels-1
            labels = labels.long()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # Calculate predictions and accuracy
            #print(outputs)
            #assert False
            #preds = torch.sigmoid(outputs)
            predicted = (outputs[:,1] > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# Assuming ONEDCNN class is defined somewhere
from ONEDCNN import ONEDCNN  # make sure you replace this with your actual import

# Hyperparameters and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


for i in range(1,6):
    # Load your model
    model = ONEDCNN().to(device)
    model.load_state_dict(torch.load(f"models/trained_split{i}", map_location=device))
    model.eval()

    # Load your datasets
    # Replace with your actual dataset and DataLoader logic
    train_dataset = RowWiseCSVLoader(f"traintest1/X_train_fold_{i}.csv", f"traintest1/y_train_fold_{i}.csv")
    test_dataset = RowWiseCSVLoader(f"traintest1/X_test_fold_{i}.csv", f"traintest1/y_test_fold_{i}.csv")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loss function
    criterion = nn.CrossEntropyLoss()



    # Run diagnostics
    train_loss, train_acc = evaluate(model, train_loader, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"{i}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
    print(f"{i}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")