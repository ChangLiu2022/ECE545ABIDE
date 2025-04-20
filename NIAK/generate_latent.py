import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CSVloader1 import RowWiseCSVLoader1
from ONEDCNN import ONEDCNN
import pandas as pd
import torch.nn.functional as F


def evaluate(model, dataloader, endDf):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for id, inputs, labels in dataloader:
            inputs = inputs.to(device)
            #labels = labels.to(device).float().view(-1, 1)  # Ensure shape compatibility

            # x = inputs.unsqueeze(1)
            # x = F.elu(model.conv1(x))
            # x = model.maxpool(x)
            # x = torch.flatten(x, 1)
            # x = model.dense_layers(x)
            
            x = inputs.unsqueeze(1)
            x = F.elu(model.conv1(x))
            x = model.maxpool(x)
            x = torch.flatten(x, 1)
            x = model.dense_layers[0](x)  # Linear(self.flattened_dim, 128)
             # Convert to float list
            x_float_list = x[0].cpu().numpy().tolist()

            new_row = pd.DataFrame([x_float_list], index=[float(id)])
            endDf = pd.concat([endDf, new_row])
    return endDf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1

for i in range(1,6):
    # Load your model
    model = ONEDCNN().to(device)
    model.load_state_dict(torch.load(f"models/trained_split{i}", map_location=device))
    model.eval()

    # Load your datasets
    # Replace with your actual dataset and DataLoader logic
    train_dataset = RowWiseCSVLoader1(f"traintest1/X_train_fold_{i}.csv", f"traintest1/y_train_fold_{i}.csv")
    test_dataset = RowWiseCSVLoader1(f"traintest1/X_test_fold_{i}.csv", f"traintest1/y_test_fold_{i}.csv")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    endDf = pd.DataFrame()
    theDF = evaluate(model, train_loader, endDf)
    theDF.to_csv(f'traintestgen128/X_train_fold_{i}.csv')
    print("evalcompletetrain")
    endDf = pd.DataFrame()
    theDF = evaluate(model, test_loader, endDf)
    theDF.to_csv(f'traintestgen128/X_test_fold_{i}.csv')
    print("evalcompletetest")
    
    

    