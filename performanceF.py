import torch

def performance(model,train_loader,test_loader,device):
    #Tests the model on all training and testing data
    #returns [train accuracy,test accuracy]
    output = [0,0]
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device).to(torch.float32)
            labels = labels.to(device)
            
            outputs = model(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()     
        output[0] = 100*correct/total
        
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _,predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        output[1] = 100*correct/total
    return(output) 