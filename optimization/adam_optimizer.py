# adam optimization algorithm with pytorch 
 # popular optimization for deep learning 
 
#--import python packages 
import numpy as np 
import torch
from torch import optim 

def adam_optimization_algo(model, lr, weight_decay):
    #input:
    # model: pytorch model (i.e. machine learning model) 
    # lr: learning rate 
    # weight_decay: weight decay in adam optimization for L2 penalty 
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    return optimizer 