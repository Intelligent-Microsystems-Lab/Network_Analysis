import json
from copy import deepcopy

#read in data and convert to dicts
storage_dir = 'Final_Dicts/'
data = ['preDict','duringDict','postDict']
for (i,element) in enumerate(data):
  with open('{}{}.json'.format(storage_dir,element),'r') as f:
    text = f.readline()
  data[i] = json.loads(text)

#reorgainize the data
fields = [i for i in data[0].keys()]
train_results = data[0][fields[0]].copy()

for key in train_results.keys():
  train_results[key] = [[] for i in range(len(data))]
 
test_results = deepcopy(train_results)

for i,item in enumerate(data):   #iterate through all stages-pre,during,post
  for cat,comb in item.items():     #iterate through all combinations-(0,1),(1,2),etc.
    for key,val in comb.items(): #iterate through all layers of the net
      if 'train' in cat:
        train_results[key][i].append(val)
      else:
        test_results[key][i].append(val)

 
#plot results
#subplot(nrows, ncols, index, **kwargs)
import matplotlib.pyplot as plt
keys = [i for i in train_results.keys() if 'Euclid' in i]
n = len(keys)
for i,key in enumerate(keys):
  plt.subplot(n,2,2*i+1)
  plt.hist(train_results[key][0],bins=100,range=[0,1])
  plt.title(f'{key} Before')
  plt.subplot(n,2,2*i+2)
  plt.hist(train_results[key][2],bins=100,range=[0,1])
  plt.title(f'{key} After')
plt.show()


#compute the Fisher test
from statistics import mean
from random import shuffle

def FisherMeans(X,Y,n):
  #inputs: X, Y-two samples stored as python lists
  #Performs the Fisher permutation test of
  #difference of means in order to estimate a p-value
  nx = len(X)
  ny = len(Y)
  
  t = abs(mean(X)-mean(Y)) #the test statistic
  U = X + Y   #pooled sample of X and Y
  Nextreme = 0 #number of samples more extreme than t
  
  #Monte-Carlo iteration
  for i in range(n):
    #randomly permute order of U
    Uprime = U.copy() 
    shuffle(Uprime)
    
    #split into Xprime and Yprime
    Xprime = Uprime[:nx]
    Yprime = Uprime[nx:]
    
    #compute the new test statistic
    tprime = abs(mean(Xprime)-mean(Yprime))
    
    #increment Nextreme if tprime>t
    if tprime > t:
      Nextreme += 1
    
  p = Nextreme/(n+1)
  
  return p

n = 1000

train_fisher = deepcopy(train_results)
test_fisher = deepcopy(test_results)
for key in train_fisher.keys():
  print(key)
  train = train_fisher[key]
  test = test_fisher[key]
  
  train_fisher[key] = FisherMeans(train[0],train[2],n)
  test_fisher[key] = FisherMeans(test[0],test[2],n)

print('Training:')
for key,val in train_fisher.items():
  print(f'{key}: {val}')
print('Testing:')
for key,val in test_fisher.items():
  print(f'{key}: {val}')
"""
#can plot the 2d PCA of two classes
from sklearn.decomposition import PCA
import torch

def plotPCA(class1,class2):
  #assume class1 and class2 are unrolled
  #both should be tensors
  combined = torch.cat((class1,class2),dim=0)
  pca = PCA(n_components=2)
  principalComponents = pca.fit_transform(combined)
  
  #slice the output
  class1_x = principalComponents[:len(class1),0]
  class1_y = principalComponents[:len(class1),1]
  class2_x = principalComponents[len(class1):,0]
  class2_y = principalComponents[len(class1):,1]
  
  #construct the plot
  plt.plot(class1_x,class1_y,'ro')
  plt.plot(class2_x,class2_y,'bo')
  plt.show()
"""