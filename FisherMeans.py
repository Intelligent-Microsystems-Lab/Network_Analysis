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