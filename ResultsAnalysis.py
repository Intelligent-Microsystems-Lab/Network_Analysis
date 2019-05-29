import json
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.decomposition import PCA

class ResultsAnalysis():
    def __init__(self,storage_dir):
        self.storage_dir = storage_dir
        self.stages = ['preDict','postDict']
        self.train_results,self.test_results = self.load_data(self.stages)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def load_data(self,data):
        #loads the JSON dicts into data and formats them correctly
        for (i,element) in enumerate(data):
            with open('{}{}.json'.format(self.storage_dir,element),'r') as f:
                text = f.readline()
            data[i] = json.loads(text)

        #reorganizes the dataset
        fields = [i for i in data[0].keys()]
        train_results = data[0][fields[0]].copy()

        for key in train_results.keys():
            train_results[key] = [[] for i in range(len(data))]
        test_results = deepcopy(train_results)

        for i,item in enumerate(data):  #iterate through all stages-pre,during,post
            for cat,comb in item.items():    #iterate through all combinations-(0,1),(1,2),etc.
                for key,val in comb.items(): #iterate through all layers of the net
                    if 'train' in cat:
                        train_results[key][i].append(val)
                    else:
                        test_results[key][i].append(val)

        return train_results,test_results

    def plot_results(self,results,stages=['Before','After'],metric='Euclid',by_layer=False):
        #plots a before and after of the results
        #results: Either train_results or test_results
        #stages: What to label each graph
        #metric: Distance function, "Euclid" for example
        #by_layer: plots all at once if False, or one layer at a time if true
        keys = [i for i in results if metric in i]
        n = len(keys)
        m = len(stages)

        if by_layer:
            for i,key in enumerate(keys):
                for j,name in enumerate(stages):
                    plt.subplot(1,m,j+1)
                    plt.hist(results[key][j],bins=100,range=[0,1])
                    plt.title('{} {}'.format(key,name))
                plt.show()

        else:
            for i,key in enumerate(keys):
                for j,name in enumerate(stages):
                    plt.subplot(n,m,m*i+j+1)
                    plt.hist(results[key][j],bins=100,range=[0,1])
                    plt.title('{} {}'.format(key,name))
            plt.show()

    def FisherMeans(self,X,Y,n,batches=1):
        #inputs: X, Y-two samples stored as python lists
        #n-number of iterations
        #Performs the Fisher permutation test of
        #difference of means in order to estimate a p-value
        nx = len(X)
        ny = len(Y)

        U = torch.Tensor(X+Y).to(self.device) #pooled sample of X and Y
        X = np.array(X)
        Y = np.array(Y)
        t = np.abs(np.mean(X)-np.mean(Y)) #test statistic
        nU = len(U)
        Nextreme = 0 #number of samples more extreme than # TEMP:

        nBatch = int(n/batches)
        for batch in range(batches):
            perms = torch.stack([torch.randperm(nU) for i in range(int(nBatch))]).to(self.device)
            shuffled = torch.stack([U]*nBatch).to(self.device)
            shuffled = shuffled.scatter(1,perms,shuffled)

            shuffled_x = shuffled[:,:nx]
            shuffled_y = shuffled[:,nx:]

            shuffled_x = torch.sum(shuffled_x,dim=1)/nx
            shuffled_y = torch.sum(shuffled_y,dim=1)/ny

            tprimes = torch.abs(shuffled_x-shuffled_y)
            tprimes = tprimes>t
            Nextreme += torch.sum(tprimes).to(torch.float32)

        p = Nextreme/(n+1)
        return p

    def FisherTraining(self,n,batches=1):
        #Compares HP values before and after training at each layer
        train_fisher = deepcopy(self.train_results)
        test_fisher = deepcopy(self.test_results)

        for key in train_fisher.keys():
            print(key)
            train = train_fisher[key]
            test = test_fisher[key]

            train_fisher[key] = self.FisherMeans(train[0],train[-1],n,batches=batches)
            test_fisher[key] = self.FisherMeans(test[0],test[-1],n,batches=batches)

        print('Training:')
        for key,val in train_fisher.items():
            print(f'{key}: {val}')
        print('Testing:')
        for key,val in test_fisher.items():
            print(f'{key}: {val}')

    def FisherLayers(self,n,batches=1):
        #Compares HP values between each layer in a network
        train_fisher = deepcopy(self.train_results)
        test_fisher = deepcopy(self.test_results)
        train_outcome = {}
        test_outcome = {}

        keys = [i for i in train_fisher.keys()]

        for i in range(len(keys)-1):
            name = f'{keys[i]} vs {keys[i+1]}'
            print(name)

            train_outcome[name] = self.FisherMeans(train_fisher[keys[i]][-1],train_fisher[keys[i+1]][-1],n,batches=batches)
            test_outcome[name] = self.FisherMeans(test_fisher[keys[i]][-1],test_fisher[keys[i+1]][-1],n,batches=batches)

        print('Training:')
        for key,val in train_outcome.items():
            print(f'{key}: {val}')
        print('Testing:')
        for key,val in test_outcome.items():
            print(f'{key}: {val}')

    def plotPCA(self,class1,class2):
        #This function combines the two given classes and performs
        #PCA on the combination, reducing down to 2 dimensions.
        #It then visualizes the two classes on a plot
        #class1 and class2 should be unrolled tensors
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


ra = ResultsAnalysis('Hadamard_Dicts/')
#ra.plot_results(ra.train_results,by_layer=True) #can comment out or swap in ra.test_results
#ra.FisherTraining(100000,batches=1)  #modify n/batches as needed
ra.FisherLayers(100000,batches=1)
