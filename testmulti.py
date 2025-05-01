import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CSVloader2 import RowWiseCSVLoader2
from multihead import LearnedQueryAttentionClassifier
from multihead1 import SelfAttentionClassifier
import pickle
main_folder = "github/ECE545ABIDE/" # if you are Chang
#main_folder = ""

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            #for d1input, d2inputs, targets in loader:
                #d1input, d2inputs, targets =  d1input.to(device), d2inputs.to(device), targets.to(device)


            *inputs, labels = batch
            # Move all inputs to device
            inputs = [inp.to(device) for inp in inputs]
            #labels = labels.to(device).float().view(-1, 1)  # Ensure shape compatibility
            labels = labels.to(device)
            outputs = model(inputs)
            
            labels = labels-1
            labels = labels.float()
            loss = criterion(torch.sigmoid(outputs), labels)
            total_loss += loss.item() * inputs[0].size(0)

            # Calculate predictions and accuracy
            #print(outputs)
            #assert False
            #preds = torch.sigmoid(outputs
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# Assuming ONEDCNN class is defined somewhere


# --- Load from file using pickle ---
with open('configs.pkl', 'rb') as f:
    things_to_test = pickle.load(f)
# Hyperparameters and device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
print("things testing:")
print(things_to_test)
print()

for thing in things_to_test:
    losses = []
    accs = []
    for i in range(1,6):
        
        # Load your model
        featuredims = list(map(int, thing["seq"].split("-")))
        folder1 = f"{main_folder}mainmodels/{thing['identifier']}{thing['seq']}-{thing['self']}-{thing['dropout']}"
        #print(folder1)
        #print(f"fold: {i-1}")
        if thing["self"]:
            model = SelfAttentionClassifier(featuredims, dropout = thing["dropout"]).to(device)
        else:
            model = LearnedQueryAttentionClassifier(featuredims, dropout = thing["dropout"]).to(device)
        model.load_state_dict(torch.load(f"{folder1}/trained_split{i}", map_location=device))
        model.eval()

        # Load your datasets
        # Replace with your actual dataset and DataLoader logic
        train_dataset = RowWiseCSVLoader2(
            f"{main_folder}traintest2/{thing['identifier']}{thing['seq']}/{thing['prefix']}train_fold_{i-1}.csv",
            f"{main_folder}NIAK/traintest1/y_train_fold_{i}.csv", featuredims.copy())
        test_dataset = RowWiseCSVLoader2(
            f"{main_folder}traintest2/{thing['identifier']}{thing['seq']}/{thing['prefix']}test_fold_{i-1}.csv",
            f"{main_folder}NIAK/traintest1/y_test_fold_{i}.csv", featuredims.copy())
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Loss function
        criterion = nn.BCELoss()

        # Run diagnostics
        train_loss, train_acc = evaluate(model, train_loader, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        #if i != 4:
        losses.append((train_loss, test_loss))
        accs.append((train_acc, test_acc))
        #print(f"{i}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%")
        #print(f"{i}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")
        #print()
    print(folder1)
    average_train_loss = sum([x[0] for x in losses]) / len(losses)
    average_test_loss = sum([x[1] for x in losses]) / len(losses)
    average_train_acc = sum([x[0] for x in accs]) / len(accs)
    average_test_acc = sum([x[1] for x in accs]) / len(accs)
    print(f"average train loss: {average_train_loss:.4f}, average train accuracy: {average_train_acc*100:.2f}%")
    print(f"average test loss: {average_test_loss:.4f}, average test accuracy: {average_test_acc*100:.2f}%")
    print()
    print(f"average train accuracy: {average_train_acc*100:.2f}%")
    print(f"average test accuracy: {average_test_acc*100:.2f}%")
    print()
    print("best fold train accuracy: ", max([x[0] for x in accs])*100)
    print("best fold test accuracy: ", max([x[1] for x in accs])*100)
    print()
    print("best fold train loss: ", min([x[0] for x in losses]))
    print("best fold test loss: ", min([x[1] for x in losses]))
    print()
    