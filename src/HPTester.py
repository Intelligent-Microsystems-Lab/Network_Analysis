import torch
import torch.nn as nn
import numpy as np
import os, itertools
from HenzePenrose import HenzePenrose
from SubsetSampler import SubsetSampler

#import line_profiler

class ModelTester():
    #@profile
    def __init__(self, training_set, test_set, classes, device,batch_size=100000, num_samples=1000):
        #training_set and test_set must be iterable
        #set batch size to 1
        self.classes = classes
        self.training_set = training_set
        self.test_set = test_set
        self.num_samples = num_samples

        #every possible combination of two classes
        self.class_combinations = [i for i in itertools.combinations(self.classes,2)]

        self.device = device

        self.batch_size = batch_size
        self.HP_euclid = HenzePenrose('euclid',device,batch_size)


    #@profile
    def compare_classes(self, model, classes, mode):
        #inputs: model-a torch nn.Module class, must contain a snapshot_forward
        #and a snapshot dict
        #classes-the two classnames to test over, should be a tuple
        #mode-either train or test

        #initialize the two dataloaders
        if mode == 'train':
            dataset = self.training_set
        else:
            dataset = self.test_set

        sampler1 = SubsetSampler(dataset,classes[0],n=self.num_samples)
        sampler2 = SubsetSampler(dataset,classes[1],n=self.num_samples)
        class_1_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size = self.batch_size,
                                                    sampler=sampler1)
        class_2_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size = self.batch_size,
                                                    sampler=sampler2)

        #run model on the data and store the outputs
        with torch.no_grad():
            model.refresh()
            for (images,labels) in class_1_loader:
                images = images.to(self.device).to(torch.float32)
                model.snapshot_forward(images)

            snapshot_1 = model.snapshot.copy()
            model.refresh()

            for (images,labels) in class_2_loader:
                images = images.to(self.device).to(torch.float32)
                model.snapshot_forward(images)

            snapshot_2 = model.snapshot.copy()
            model.refresh()

        #snapshots 1 and 2 should now contain dicts with the results at each stage from classes 1 and 2
        scores = {}

        for key in snapshot_1.keys():
            class_1_item = torch.cat(snapshot_1[key])
            class_2_item = torch.cat(snapshot_2[key])

            scores['Euclid '+key] = self.HP_euclid(class_1_item,class_2_item)

        return scores


    #@profile
    def __call__(self, model):
        #input: model
        #output: a dict of dicts with all the test results for the entire model
        data = {}

        for combo in self.class_combinations:
            print(combo)
            data[str(combo)+' train'] = self.compare_classes(model,combo,'train')
            torch.cuda.empty_cache()
            data[str(combo)+' test'] = self.compare_classes(model,combo,'test')
            torch.cuda.empty_cache()

        return(data)
