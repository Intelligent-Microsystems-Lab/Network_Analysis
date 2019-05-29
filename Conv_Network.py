import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #sets up parameters for convolutional layers
        self.conv1 = nn.Conv2d(1, 32,kernel_size=3, stride=1,padding=3)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3, stride=1,padding=1)

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(2)

        self.refresh()
        self.cpu = torch.device('cpu')

    def refresh(self):
        self.snapshot = {'initial':[],
                         'conv1':[],
                         'relu1':[],
                         'maxpool1':[],
                         'conv2':[],
                         'relu2':[],
                         'maxpool2':[],
                         'fc1':[],
                         'relu3':[],
                         'fc2':[],
                         'softmax':[]}
        torch.cuda.empty_cache()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = out.view(-1,4096)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return(out)

    def snapshot_forward(self,x):
        self.snapshot['initial'].append(x.clone().detach().to(self.cpu))

        out = self.conv1(x)
        self.snapshot['conv1'].append(out.clone().detach().to(self.cpu))

        out = self.relu(out)
        self.snapshot['relu1'].append(out.clone().detach().to(self.cpu))

        out = self.maxpool(out)
        self.snapshot['maxpool1'].append(out.clone().detach().to(self.cpu))

        out = self.conv2(out)
        self.snapshot['conv2'].append(out.clone().detach().to(self.cpu))

        out = self.relu(out)
        self.snapshot['relu2'].append(out.clone().detach().to(self.cpu))

        out = self.maxpool(out)
        self.snapshot['maxpool2'].append(out.clone().detach().to(self.cpu))

        out = out.view(-1,4096)
        out = self.fc1(out)
        self.snapshot['fc1'].append(out.clone().detach().to(self.cpu))

        out = self.relu(out)
        self.snapshot['relu3'].append(out.clone().detach().to(self.cpu))

        out = self.fc2(out)
        self.snapshot['fc2'].append(out.clone().detach().to(self.cpu))

        out = self.softmax(out)
        self.snapshot['softmax'].append(out.clone().detach().to(self.cpu))

        return(out)
