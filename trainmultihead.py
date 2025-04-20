import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CSVloader2 import RowWiseCSVLoader2 # for seq 128:182
from multihead import LearnedQueryAttentionClassifier
from multihead1 import SelfAttentionClassifier
import os
import pickle
#main_folder = "github/ECE545ABIDE/" # if you are Chang
main_folder = ""


with open('configs.pkl', 'rb') as f:
    configurations = pickle.load(f)

print("beginning training:")
print("configurations:")
print(configurations)
print()

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
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 1e-2)


        # Dataset and DataLoader
        dataset = RowWiseCSVLoader2(
            f"{main_folder}traintest2/{config['identifier']}{config['seq']}/{config['prefix']}train_fold_{i-1}.csv",
            f"{main_folder}NIAK/traintest1/y_train_fold_{i}.csv", featuredims.copy())
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Training loop
        n_epochs = 100
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

            if ((epoch+1) % 5 == 0):
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(loader):.4f}")
            
        folder1 = f"{main_folder}mainmodels/{config['identifier']}{config['seq']}-{config['self']}-{config['dropout']}"
        os.makedirs(folder1, exist_ok=True)
        torch.save(model.state_dict(), f"{folder1}/trained_split{i}")
        print()