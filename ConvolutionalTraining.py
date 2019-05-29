import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

#importing custom modules
from Conv_Network import CNN
from performanceF import performance

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
num_epochs = 200
num_classes = 10
batch_size = 10
learning_rate = .001

model_storage_dir = 'Conv_Parameters/'
if not os.path.isdir(model_storage_dir):
	os.mkdir(model_storage_dir)


#loads MNIST
train_dataset = torchvision.datasets.MNIST(root='./MNISTdata', train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./MNISTdata', train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#load in the model from FC_Network.py
model = CNN().to(device).to(torch.float32)

#setting up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#tracker definition
loss_tracker = []
train_accuracy = []
test_accuracy = []

#begin training
total_step = len(train_loader)
for epoch in range(num_epochs):
    if epoch % 99 == 0:
	    a = model.state_dict()
	    print(a.keys())
	    torch.save(model.state_dict(), model_storage_dir+'conv'+str(epoch)+'.ckpt')

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device).to(torch.float32)
        labels = labels.to(device)

        #forward propogation and loss calculation
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_tracker.append(loss.item())

        #backprop and weight adjustment
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #progress report at fixed intervals
        if i % 1000 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {}'.format(
                    epoch+1, num_epochs, i, total_step, loss.item()))

    accuracies = performance(model,train_loader,test_loader,device)
    train_accuracy.append(accuracies[0])
    test_accuracy.append(accuracies[1])
    np.save('Train_Accuracy.npy',np.array(train_accuracy))
    np.save('Test_Accuracy.npy',np.array(test_accuracy))
    np.save('error_tracker.npy', np.array(loss_tracker))




model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device).to(torch.float32)
        labels = labels.to(device)

        outputs = model(images)
        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
