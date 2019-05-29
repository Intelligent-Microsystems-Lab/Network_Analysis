import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
import json,os
#import line_profiler

from HPTester import ModelTester	#code to compute the HP statistics necessary
from FC_Network import FC			#the neural network under test

#run computations on the GPU if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')			

#where the parameters have been saved
model_storage_dir = 'FC_Parameters/'
result_storage_dir = 'Final_Dicts/'
if not os.path.isdir(result_storage_dir):
	os.mkdir(result_storage_dir)

#construct the datasets
train_dataset = torchvision.datasets.MNIST(root='./MNISTdata', train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./MNISTdata', train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)

practice_loader1 = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=1,
                                          shuffle=True)
										  
practice_loader2 = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)

#initialize the model three times to each of the three different parameters
model_pre = FC()
model_during = FC()
model_post = FC()

#load in all generated parameters
model_pre.load_state_dict(torch.load(model_storage_dir+'fc0.ckpt'))
model_pre.to(device).to(torch.float32)
    
model_during.load_state_dict(torch.load(model_storage_dir+'fc99.ckpt'))
model_during.to(device).to(torch.float32)
    
model_post.load_state_dict(torch.load(model_storage_dir+'fc198.ckpt'))
model_post.to(device).to(torch.float32)
			
#run the HP testing for each model			
testing = ModelTester(practice_loader1,practice_loader2,[i for i in range(10)],'./Transformed Data/',device)

pre_results = testing(model_pre)
during_results = testing(model_during)
post_results = testing(model_post)

#store the results from testing in json files
with open(result_storage_dir+'preDict.json','w') as f:
	json.dump(pre_results, f)
	
with open(result_storage_dir+'duringDict.json','w') as f:
	json.dump(during_results, f)
	
with open(result_storage_dir+'postDict.json','w') as f:
	json.dump(post_results, f)

#to reclaim the dictionaries:
#with open(filename) as f:
#	my_dict = json.load(f)
	