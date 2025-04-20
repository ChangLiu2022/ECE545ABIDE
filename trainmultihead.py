import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from CSVloader2 import RowWiseCSVLoader2
from multihead import LearnedQueryAttentionClassifier

for i in range(1,2):
    # Assuming ONEDCNN class is defined as you posted
    model = LearnedQueryAttentionClassifier([128,182])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 1e-2)


    # Dataset and DataLoader
    dataset = RowWiseCSVLoader2(f"traintest2/combined_v2_X2_train_fold_{i-1}.csv", f"NIAK/traintest1/y_train_fold_{i}.csv")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Training loop
    n_epochs = 100
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for d1input, d2inputs, targets in loader:
            d1input, d2inputs, targets =  d1input.to(device), d2inputs.to(device), targets.to(device)

            #targets = targets-1
            #targets = targets.long()
            # Forward pass
            
            outputs = model([d1input, d2inputs])
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

        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {running_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), f"mainmodels/trained_split{i}")