import torch
import torch.nn as nn
import numpy as np
import os, itertools
from HenzePenrose import HenzePenrose
from SubsetSampler import SubsetSampler

#import line_profiler

class ModelTester():
    #@profile
    def __init__(self, training_set, test_set, classes, save_dir, device,batch_size=100000):
        #training_set and test_set must be iterable
        #set batch size to 1
        self.classes = classes
        self.save_dir = save_dir
        self.training_set = training_set
        self.test_set = test_set

        """
        if not os.path.isdir(save_dir):
            print('Reorganizing the dataset')
            os.mkdir(save_dir)
            self.dataset_to_numpy(training_set,'train')
            self.dataset_to_numpy(test_set,'test')
        else:
            print('Dataset present at this directory')
        """

        #every possible combination of two classes
        self.class_combinations = [i for i in itertools.combinations(self.classes,2)]

        self.device = device

        self.batch_size = batch_size
        self.HP_euclid = HenzePenrose('euclid',device,batch_size)
        #self.HP_cosine = HenzePenrose('cosine')

    """
    #@profile
    def dataset_to_numpy(self, dataset, mode):
        #inputs: dataset-iterable with the data
        #mode-the extension to distinguish which loader it is-a string
        for classname in self.classes:
            data = None
            for i, (image, labels) in enumerate(dataset):
                if labels.item() == classname:
                    if data is not None:
                        data = torch.cat((data,image), dim=0)
                    else:
                        data = image

            data = data.numpy()
            np.save(self.save_dir+mode+str(classname)+'.npy', data)
            print(mode+' '+str(classname))
    """
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

        sampler1 = SubsetSampler(dataset,classes[0],n=1000)
        sampler2 = SubsetSampler(dataset,classes[1],n=1000)
        class_1_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size = self.batch_size,
                                                    sampler=sampler1)
        class_2_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size = self.batch_size,
                                                    sampler=sampler2)

        #run model on the data and store the outputs
        with torch.no_grad():
            model.refresh()
            for i, (images,labels) in enumerate(class_1_loader):
                images = images.to(self.device).to(torch.float32)
                model.snapshot_forward(images)

            snapshot_1 = model.snapshot.copy()
            model.refresh()

            for i, (images,labels) in enumerate(class_2_loader):
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


        """
        class_1_data = torch.Tensor(np.load(self.save_dir+mode+str(classes[0])+'.npy')).to(self.device)
        class_2_data = torch.Tensor(np.load(self.save_dir+mode+str(classes[1])+'.npy')).to(self.device)

        #class_1_data = class_1_data[:100]
        #class_2_data = class_2_data[:100]
        batches1 = int(np.ceil(len(class_1_data)/self.batch_size))
        for i in range(batches1):
                if (i+1)*self.batch_size > len(class_1_data):
                        model.snapshot_forward(class_1_data[i*self.batch_size:])
                else:
                        model.snapshot_forward(class_1_data[i*self.batch_size:(i+1)*self.batch_size])
                torch.cuda.empty_cache()
        snapshot_1 = model.snapshot.copy()
        for key,val in snapshot_1.items():
                snapshot_1[key] = torch.cat(val,dim=0)

        batches2 = int(np.ceil(len(class_2_data)/self.batch_size))
        for i in range(batches2):
                if (i+1)*self.batch_size > len(class_2_data):
                        model.snapshot_forward(class_2_data[i*self.batch_size:])
                else:
                        model.snapshot_forward(class_2_data[i*self.batch_size:(i+1)*self.batch_size])
                torch.cuda.empty_cache()
        snapshot_2 = model.snapshot.copy()
        for key,val in snapshot_2.items():
                snapshot_2[key] = torch.cat(val,dim=0)

        #snapshots 1 and 2 should now contain dicts with the results at each stage from classes 1 and 2
        scores = {}

        for key in snapshot_1.keys():
            class_1_item = snapshot_1[key]
            class_2_item = snapshot_2[key]

            scores['Euclid '+key] = self.HP_euclid(class_1_item,class_2_item)
            #scores['Cosine '+key] = self.HP_cosine.calculateHP(class_1_item,class_2_item)


        #print(scores['Euclid Initial'])
        return(scores)
    """
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
