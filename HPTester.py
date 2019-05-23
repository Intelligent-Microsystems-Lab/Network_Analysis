import torch
import torch.nn as nn
import numpy as np
import os, itertools
from HenzePenrose import HenzePenrose

import line_profiler

class ModelTester():
  @profile
  def __init__(self, training_set, test_set, classes, save_dir, device):
    #training_set and test_set must be iterable
    #set batch size to 1
    self.classes = classes
    self.save_dir = save_dir
    
    if not os.path.isdir(save_dir):
      print('Reorganizing the dataset')
      os.mkdir(save_dir)
      self.dataset_to_numpy(training_set,'train')
      self.dataset_to_numpy(test_set,'test')
    else:
      print('Dataset present at this directory')
    
    #every possible combination of two classes
    self.class_combinations = [i for i in itertools.combinations(self.classes,2)]
    
    self.device = device
    
    self.HP_euclid = HenzePenrose('euclid')
    #self.HP_cosine = HenzePenrose('cosine')
  
  @profile
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
  
  @profile
  def compare_classes(self, model, classes, mode):
    #inputs: model-a torch model class, must contain a snapshot_forward
    #and a snapshot dict
    #classes-the two classnames to test over, should be a tuple
    #mode-either train or test
    class_1_data = torch.Tensor(np.load(self.save_dir+mode+str(classes[0])+'.npy')).to(self.device)
    class_2_data = torch.Tensor(np.load(self.save_dir+mode+str(classes[1])+'.npy')).to(self.device)
    
    class_1_data = class_1_data[:10]
    class_2_data = class_2_data[:10]
    
    model.snapshot_forward(class_1_data)
    snapshot_1 = model.snapshot.copy()

    model.snapshot_forward(class_2_data)
    snapshot_2 = model.snapshot.copy()
    
    #snapshots 1 and 2 should now contain dicts with the results at each stage from classes 1 and 2
    scores = {}
    
    for key in snapshot_1.keys():
      class_1_item = snapshot_1[key]
      class_2_item = snapshot_2[key]
      
      scores['Euclid '+key] = self.HP_euclid.calculateHP(class_1_item,class_2_item)
      #scores['Cosine '+key] = self.HP_cosine.calculateHP(class_1_item,class_2_item)
    
    #print(scores['Euclid Initial'])
    return(scores)
  
  @profile
  def evaluate_model(self, model):
    #input: model
    #output: a dict of dicts with all the test results for the entire model
    data = {}
    
    for combo in self.class_combinations:
      print(combo)
      data[str(combo)+' train'] = self.compare_classes(model,combo,'train')
      
      data[str(combo)+' test'] = self.compare_classes(model,combo,'test')
    
    return(data)