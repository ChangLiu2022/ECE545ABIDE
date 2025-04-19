import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ONEDCNN import ONEDCNN
from CSVloader import RowWiseCSVLoader


for i in range(1,6):
    # Assuming ONEDCNN class is defined as you posted
    model = ONEDCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Dataset and DataLoader
    dataset = RowWiseCSVLoader(f"traintest1/X_train_fold_{i}.csv", f"traintest1/y_train_fold_{i}.csv")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training loop
    n_epochs = 15
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

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

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), f"models/trained_split{i}")