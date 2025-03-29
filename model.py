
import torch
import torch.nn as nn
from NIAK.ONEDCNN import *
from sMRI.model import *

class modelSuper(nn.Module):
    def __init__(self, input_size):
        super(modelSuper, self).__init__()
        self.ONEDCNN = ONEDCNN() # 19900, 1, 2 by default
        self.TWODCNN = ASDModelBinary()
        self.THEEDCNN = nn.Linear(1, 1) #TODO replace this model with the 3DCNN model
        self.GNN = nn.Linear(1, 1) #TODO replace this model with the GNN model
    
    def forward(self, x):
        # x is a tuple (SMRI, CPAC, NIAK) 
        
        
        
        
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out = 0
        return out
    
    def get_name(self):
        return "MainModel"