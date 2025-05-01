import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ONEDCNN import ONEDCNN
from CSVloader import RowWiseCSVLoader

#main_folder = "github/ECE545ABIDE/NIAK/" # if you are Chang
main_folder = ""



BATCH_SIZE = 16
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            #labels = labels.to(device).float().view(-1, 1)  # Ensure shape compatibility
            labels = labels.to(device)
            outputs = model(inputs)
            
            labels = labels-1
            labels = labels.long()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # Calculate predictions and accuracy
            #print(outputs)
            #assert False
            #preds = torch.sigmoid(outputs)
           # print("OUTS")
           # print(outputs[:,1])
            predicted = (outputs[:,1] > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

#stds = [0.01, 0.05, 0.1, 0.2]
stds = [0.3, 0.4, 0.5, 0.6]
for s1 in stds:
    print("DOING RUN ON", s1)
    for i in range(1,2):
        # Assuming ONEDCNN class is defined as you posted
        model = ONEDCNN(std = 0.0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        test_dataset = RowWiseCSVLoader(f"{main_folder}traintest1/X_test_fold_{i}.csv", f"{main_folder}traintest1/y_test_fold_{i}.csv")
        #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Dataset and DataLoader
        dataset = RowWiseCSVLoader(f"{main_folder}traintest1/X_train_fold_{i}.csv", f"{main_folder}traintest1/y_train_fold_{i}.csv")
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Training loop
        n_epochs = 25
        model_state = model.state_dict()
        test_acc1 = 0
        print("starting training")
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)

                targets = targets-1
                targets = targets.long()
                # Forward pass
                
                outputs = model(inputs)
                #print (outputs.shape)
                #print(targets.shape)
                #assert False
                loss = criterion(outputs, targets)

                # Apply L1 regularization to dense_layers[4]
                #print(model.dense_layers[4].parameters())
                l1_params = model.dense_layers[4].parameters()
                l1_loss = model.l1_strength * sum(p.abs().sum() for p in l1_params)

                # Combine losses
                total_loss = loss + l1_loss

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

                #targets = targets-1
                #targets = targets.long()
                
                predicted = (outputs[:,1] > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(loader):.4f}, acc : {correct/total}")
            test_l,test_acc = evaluate(model, test_loader, criterion)
            print(f"        Test Loss: {test_l:.4f}, Test Accuracy: {test_acc:.4f}")
            if test_acc > test_acc1:
                test_acc1 = test_acc
                print("saving new best model")
                model_state = model.state_dict()
        
        #torch.save(model.state_dict(), f"{main_folder}modelsrnd2/trained_split{i}")
        torch.save(model_state, f"{main_folder}modelsrnd2/{s1}trained_split{i}")