# Henze-Penrose Network Profiler
Based off of: 
*Inside the Black Box: Characterizing Deep Learning Functional Mappings*
https://arxiv.org/pdf/1907.04223.pdf

Written for Python 3.x and Pytorch


### Conceptual Introduction
The goal of this library is to allow users to gain more insight into 
what goes on inside their neural networks during training. This is
accomplished by analyzing the *class separation* in a neural netowork.
Class separation is quantified by the Henze-Penrose statistic, which
is found by constructing the minimum spanning tree between data of two
classes, and finding how many edges connect the same class vs. different 
classes. 

After defining this Henze-Penrose statistic, we can then use it to gain 
a deeper understanding of model performance. This can be done in multiple
ways, but we implement it here by comparing each class to each other 
class (in a dataset, numbers in MNIST for example), and examining this
at the output of each layer. 

This data can then be analyzed in a couple of ways. Here, we implement
two useful methods: distribution graphing in Matplotlib and the Fischer
Test for Means. With these tools, you can compare separation before and
after a layer of a neural net to see how that layer transforms the data,
or compare separation before and after training. High separation values
(close to 1) indicate high separation, and low separation values (close
to 0) indicate less separation. In general, you can gain insight into 
the effect a layer is having by seeing if it increases separation, and 
you can see how a model is learning over time by checking if class 
separation increases before and after training.


### Library Usage
A couple steps need to be taken in order to analyze your networks with
this tool. To follow along with this guide, view the `example` folder.

#### Step 1: Network Definition
The first step is to define your network in the standard Pytorch way,
by inheriting from the `nn.Module` class and defining a `forward` method.
In addition to the standard methods employed, two new methods should be
written, namely `refresh()` and `snapshot_forward()`.

The network analyzer keeps track of the output of each layer, and does so
through a `dict` called `snapshot`. Your `refresh()` method should take
a form similar to the following:

``` Python
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
```

The purpose of this method is to initialize the `dict` between samples, while
also ensuring that any GPUs utilized do not run out of memory. When writing
your own method, include an entry for each point in the network you wish to 
track.

The `snapshot_forward` method is basically a copy of your network's standard
forward pass definition. The only difference is that it copies the output at 
each layer and saves it to the `snapshot`. Structure your method similar to the
following example. A useful tip is to implement some sort of processing function
to ensure that the output at a given stage is completely separated from the 
computational graph. The `process` lambda function shown below works for me.
``` Python
# self.process = lambda x: x.clone().detach().to(self.cpu)
def snapshot_forward(self,x):
    self.snapshot['initial'].append(self.process(x))

    out = self.conv1(x)
    self.snapshot['conv1'].append(self.process(out))

    out = self.relu(out)
    self.snapshot['relu1'].append(self.process(out))

    out = self.maxpool(out)
    self.snapshot['maxpool1'].append(self.process(out))

    out = self.conv2(out)
    self.snapshot['conv2'].append(self.process(out))

    out = self.relu(out)
    self.snapshot['relu2'].append(self.process(out))

    out = self.maxpool(out)
    self.snapshot['maxpool2'].append(self.process(out))

    out = out.view(-1,4096)
    out = self.fc1(out)
    self.snapshot['fc1'].append(self.process(out))

    out = self.relu(out)
    self.snapshot['relu3'].append(self.process(out))

    out = self.fc2(out)
    self.snapshot['fc2'].append(self.process(out))

    out = self.softmax(out)
    self.snapshot['softmax'].append(self.process(out))

    return(out)
```

A full example of how this could be implemented for a CNN can be
found in `./example/Conv_Network.py`

#### Step 2: Network Training
Training your network is largely the same as any Pytorch training
routine. The only difference is that you should save your network
weights at every instance you would like to analyze later. For 
example, you could add the following code to your training loop
in order to save weights every 100 epochs.

``` Python
if epoch % 100 == 0:
    torch.save(model.state_dict(), f'{model_storage_dir}conv{epoch}.ckpt')
```

A full example of how this could be implemented for a CNN can be
found in `./example/ConvolutionalTraining.py`

#### Step 3: Interfacing with the Library
After training has been concluded, we can interface with the library
to run an analysis on the network. A full example for how to do this
for a CNN can be found in `./example/HP_CNN_Wrapper.py`, and I will
walk through that example below.

##### a) Import Libraries and Set Parameters
This step is fairly standard, and just involves importing all the 
necessary libraries.

``` Python
import torch
import torch.nn
import torchvision
import torchvision.transforms as transforms
import json,os
#import line_profiler

from HPTester import ModelTester	#code to compute the HP statistics necessary
from Conv_Network import CNN			#the neural networks under test

#run computations on the GPU if possible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#where the parameters have been saved
model_storage_dir = 'Conv_Parameters/'
result_storage_dir = 'Conv_Dicts/'
if not os.path.isdir(result_storage_dir):
	os.mkdir(result_storage_dir)
```

##### b) Constructing the Datasets
This library relies heavily on the Pytorch `dataset` framework. This 
works very well for simple models like MNIST which come preloaded in
Pytorch, but you may have to write your own dataset wrapper if using
a custom dataset. Instructions for doing so can be found [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

For a simple MNIST application, use the following:
``` Python
construct the datasets
train_dataset = torchvision.datasets.MNIST(root='./MNISTdata', train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./MNISTdata', train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)
```

##### c) Initializing Models
We must load in each model we plan to analyze and initialize them to 
their training weights. This can be done as follows:

``` Python
#initialize the models
model_pre = CNN()
model_post = CNN()

#load in all generated parameters
model_pre.load_state_dict(torch.load(model_storage_dir+'conv0.ckpt'))
model_pre.to(device).to(torch.float32)

model_post.load_state_dict(torch.load(model_storage_dir+'conv198.ckpt'))
model_post.to(device).to(torch.float32)
```

##### d) Calculating the HP Statistics
To do the Henze-Penrose analysis, we utilize the `ModelTester` class in the
HPTester.py file. This class is initialized with several arguments:
* `training_set`-see part c
* `test_set`-see part c
* `classes`-a list of all class labels
    * For MNIST, this would be the numbers 0-9
* `device`-A `torch.device` object, see part a for initialization
    * A GPU is vastly preferred, but it should work on CPU as well
* `batch_size`-the number of items to analyze at once
    * Play around with different numbers to see what works best for you
* `num_samples`-the number of samples from each class to analyze
    * Runtime increases significantly with more samples, but the 
    output will be more accurate

Once the `ModelTester` is initialized, simply call it on your models,
and it will output the class separations for the model by class in a dict. 
This dict can then be parsed and analyzed in a future step.

An example of the steps of the above can be seen below:
``` Python
#run the HP testing for each model
testing = ModelTester(train_dataset,test_dataset,[i for i in range(10)],
                      device,batch_size = 100,num_samples=1000)

pre_results = testing(model_pre)
post_results = testing(model_post)

#store the results from testing in json files
with open(result_storage_dir+'preDict.json','w') as f:
	json.dump(pre_results, f)


with open(result_storage_dir+'postDict.json','w') as f:
	json.dump(post_results, f)
```

##### e) Analyzing the Results
Due to the relatively large amount of time it takes to calculate
the Henze-Penrose statistics for each combination of classes,
the analysis of the results is kept in its own module, `ResultsAnalysis.py`.

In this file, I define the `ResultsAnalysis` class. It contains several
useful methods which are illustrated below. Feel free to modify existing
methods or create your own, any extra functionality would be welcome.

* `__init__`-the initialization takes two arguments: 
    * storage_dir: the path to the saved output
        * `Conv_Dicts/` in our current example
    * stages: a list of all the stage names
        * `['preDict','postDict']` in the example we've been following
* `load_data`-called automatically, loads and parses the data stored as JSON dicts, takes one argument:
    * data: should be the same as the `stages` argument in `__init__`
* `plot_results`-plots the results as a series of Pyplot histograms, takes several parameters
    * results: the loaded data, get with `ra.train_results` or `ra.test_results`, where `ra` is the name of the object
    * stages: a list of labels for each stage, defaults to `['Before','After']`
    * metric: defaults to `Euclid`, change if a different distance metric was used
    * by_layer: either True or False. Can either get each layer individually or all at once
* `FisherTraining`-uses the Fisher permutation test for means to analyze class separation before and after training a network. This model is probabilistic and relies on GPU acceleration. It uses the object's internal train_results and test_results, and takes two parameters:
    * n: the number of Monte Carlo similations to run
    * batches: the number of batches to use-GPU optimized
* `FisherLayers`-uses the Fisher permutation test for means to analyze class separation between layers in a network. This model is probabilistic and relies on GPU acceleration. It uses the object's internal train_results and test_results, and takes two parameters:
    * n: the number of Monte Carlo similations to run
    * batches: the number of batches to use-GPU optimized
* `plotPCA`-can be used to reduce a high-dimensional input to a 2D output. Meant as a sanity check for the rest of the high-dimensional embedding tools. Only takes 2 inputs, which are stored as Pytorch tensors.

More tools may be added to this module in the future.