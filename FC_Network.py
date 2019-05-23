import torch
import torch.nn as nn

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()    
        self.fc1 = nn.Linear(784, 96)
        self.fc2 = nn.Linear(96, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.snapshot = {}
        
    def forward(self, x):
        out = x.view(-1, 784)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return(out)
                
    def snapshot_forward(self, x):
		#runs forward and logs all values to snapshot
        out = x.view(-1,784)
        self.snapshot['Initial'] = out.clone().detach()
        
        out = self.fc1(out)
        self.snapshot['FC1'] = out.clone().detach()
        out = self.relu(out)
        self.snapshot['ReLU1'] = out.clone().detach()
        
        out = self.fc2(out)
        self.snapshot['FC2'] = out.clone().detach()
        
        out = self.softmax(out)
        self.snapshot['Softmax'] = out.clone().detach()
        return(out)