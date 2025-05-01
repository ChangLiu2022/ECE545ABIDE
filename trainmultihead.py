import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CSVloader2 import RowWiseCSVLoader2 # for seq 128:182
from multihead import LearnedQueryAttentionClassifier
from multihead1 import SelfAttentionClassifier
import os
import pickle
main_folder = "github/ECE545ABIDE/" # if you are Chang
#main_folder = ""


with open('configs.pkl', 'rb') as f:
    configurations = pickle.load(f)

print("beginning training:")
print("configurations:")
print(configurations)
print()

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

for config in configurations:
    print("training:")
    print(config)
    print()
    
    for i in range(1,6):
        print(f"training fold {i-1}")
        # Assuming ONEDCNN class is defined as you posted
        featuredims = list(map(int, config["seq"].split("-")))
        
        if config["self"]:
            model = SelfAttentionClassifier(featuredims, dropout = config["dropout"])
        else:
            model = LearnedQueryAttentionClassifier(featuredims, dropout = config["dropout"])
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
        #lr 0.00001 weight decay 0.001 54.5 average acc for all folds


        # Dataset and DataLoader
        dataset = RowWiseCSVLoader2(
            f"{main_folder}traintest2/{config['identifier']}{config['seq']}/{config['prefix']}train_fold_{i-1}.csv",
            f"{main_folder}NIAK/traintest1/y_train_fold_{i}.csv", featuredims.copy())
        test_dataset = RowWiseCSVLoader2(
            f"{main_folder}traintest2/{config['identifier']}{config['seq']}/{config['prefix']}test_fold_{i-1}.csv",
            f"{main_folder}NIAK/traintest1/y_test_fold_{i}.csv", featuredims.copy())
        loader = DataLoader(dataset, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        # Training loop
        n_epochs = 250
        max_accuracy = 0
        modelstate = model.state_dict()
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0

            for batch in loader:
            #for d1input, d2inputs, targets in loader:
                #d1input, d2inputs, targets =  d1input.to(device), d2inputs.to(device), targets.to(device)


                *inputs, targets = batch
                # Move all inputs to device
                inputs = [inp.to(device) for inp in inputs]
                # Move labels to device
                targets = targets.to(device)
                #targets = targets-1
                #targets = targets.long()
                # Forward pass
                
                outputs = model(inputs)
                #print (outputs.shape)
                #print(targets.shape)
                #assert False
                targets = targets.float() - 1
                
                loss = criterion(torch.sigmoid(outputs), targets)
                
                # Combine losses
                total_loss = loss

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            
            #if test_acc > max_accuracy:
            #    max_accuracy = test_acc
            #    modelstate = model.state_dict()
            #    print("saving model")
            if ((epoch+1) % 5 == 0):
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(loader):.4f}, test loss: {test_loss:.4f}, test accuracy: {test_acc*100:.2f}%")
            
        folder1 = f"{main_folder}mainmodels/{config['identifier']}{config['seq']}-{config['self']}-{config['dropout']}"
        os.makedirs(folder1, exist_ok=True)

        torch.save(model.state_dict(), f"{folder1}/trained_split{i}")
        #torch.save(modelstate, f"{folder1}/trained_split{i}")
        print()