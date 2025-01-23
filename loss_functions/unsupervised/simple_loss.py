#--import python packages 

import numpy as np 
import torch 
import torch.nn as nn 

def simple_loss_algo(X_prime, X):
    # this is an example code for mean square error loss 
    loss = nn.MSELoss(reduction='mean')
    recon_loss = loss(X_prime, X) 
    return recon_loss 