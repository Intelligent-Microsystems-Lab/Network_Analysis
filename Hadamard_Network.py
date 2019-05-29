import torch
import torch.nn as nn
import numpy as np

class HadamardConv2d(nn.Module):
    def __init__(self,device):
        super(HadamardConv2d, self).__init__()
        self.device = device

        #sets up parameters for convolutional layers
        self.transform_matrix1 = torch.from_numpy(self.HadamardMatrix(5))
        self.transform_matrix2 = torch.from_numpy(self.HadamardMatrix(4))

        self.filter1 = nn.Parameter(torch.randn(32,3, 3))
        self.bias1   = nn.Parameter(torch.randn(32, 1))
        self.filter2 = nn.Parameter(torch.randn(64,3, 3))
        self.bias2   = nn.Parameter(torch.randn(64, 1))

        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.maxpool = nn.MaxPool2d(2)

        self.snapshot = {}

        self.refresh()
        self.cpu = torch.device('cpu')

    #resets the snapshot
    def refresh(self):
        self.snapshot = {'initial':[],
                         'hadamard1':[],
                         'relu1':[],
                         'maxpool1':[],
                         'hadamard2':[],
                         'relu2':[],
                         'maxpool2':[],
                         'fc1':[],
                         'relu3':[],
                         'fc2':[],
                         'softmax':[]}
        torch.cuda.empty_cache()

    #overrides the .to() function to cast default parameters to whatever is required
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.transform_matrix1 = self.transform_matrix1.to(*args, **kwargs)
        self.transform_matrix2 = self.transform_matrix2.to(*args, **kwargs)
        self.filter1 = self.filter1.to(*args, **kwargs)
        self.bias1 = self.bias1.to(*args, **kwargs)
        self.filter2 = self.filter2.to(*args, **kwargs)
        self.bias2 = self.bias2.to(*args, **kwargs)

        return self

    #implementation of a convolution layer with the hadamard transform-F(a*b)=F(a).F(b)
    #with the hadamard transform acting as a fourier transform
    #hadamard transform of a pytorch tensor
    def HadamardMatrix(self, n):
        #2^n Hadamard numpy array
        if n == 0:
            return(np.array([[1]]))
        prev = self.HadamardMatrix(n-1)
        new = np.zeros((2**n,2**n))
        half = int(new.shape[0]/2)
        new[:half,:half] = prev
        new[:half,half:] = prev
        new[half:,:half] = prev
        new[half:,half:] = -prev
        return(new/2**.5)

    def HadamardConv(self, inputs, filters, bias, transform_matrix):
        """
        shape of inputs: batch size x previous channels x image len x image width
        shape of data: batch size x previous channels x image len x image width x 2 -fft data

        shape of filters: num_filters x filter len x filter width --input filters
        skape of kernels: num_filters x image len x image width x 2 --fft filters

        shape of output: batch size x new channels x image len x image width
        """
        dim_size = transform_matrix.shape[0]

        kernels = torch.zeros(filters.shape[0],dim_size,dim_size).to(self.device).to(torch.float32)
        kernels[:,:(filters.shape[1]),:(filters.shape[2])] = filters
        kernels = torch.matmul(transform_matrix, torch.matmul(kernels, transform_matrix))

        data = torch.zeros(inputs.shape[0], inputs.shape[1], dim_size,dim_size).to(self.device).to(torch.float32)
        data[:,:,:inputs.shape[2], :inputs.shape[3]] = inputs
        data = torch.matmul(transform_matrix, torch.matmul(data, transform_matrix))

        bias = bias[:,0] #just to make compatible with previous training

        results = data.unsqueeze(2)
        results = results.expand(results.shape[0],results.shape[1],kernels.shape[0],results.shape[3],results.shape[4])
        results = torch.mul(results, kernels)
        results = torch.matmul(transform_matrix, torch.matmul(results, transform_matrix))
        results = torch.sum(results,dim=1)
        results += bias.unsqueeze(1).unsqueeze(2).expand(results.shape[1],results.shape[2],results.shape[3])

        return(results)

    def forward(self, x):
        out = self.HadamardConv(x, self.filter1, self.bias1, self.transform_matrix1)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.HadamardConv(out, self.filter2, self.bias2, self.transform_matrix2)
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
        out = self.HadamardConv(x, self.filter1, self.bias1, self.transform_matrix1)
        self.snapshot['hadamard1'].append(out.clone().detach().to(self.cpu))

        out = self.relu(out)
        self.snapshot['relu1'].append(out.clone().detach().to(self.cpu))

        out = self.maxpool(out)
        self.snapshot['maxpool1'].append(out.clone().detach().to(self.cpu))

        out = self.HadamardConv(out, self.filter2, self.bias2, self.transform_matrix2)
        self.snapshot['hadamard2'].append(out.clone().detach().to(self.cpu))

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
