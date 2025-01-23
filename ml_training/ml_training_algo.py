import numpy as np
import copy 
import torch 
#---customized functions and classes 
from optimization.adam_optimizer import adam_optimization_algo
from loss_functions.unsupervised.simple_loss import simple_loss_algo

class AE:
    def __init__(self, train_data_loader, train_data, params_hm):
        self.train_data_loader = train_data_loader
        self.train_data = train_data 
        self.params_hm = params_hm
        self.device = self.params_hm['device'] 
        
        self._model_setup() 
        
        
    def _model_setup(self):
        # instantiate/call your neural architecture here 
        self.model = 3 
        
        
    def fit(self):
        opti = adam_optimization_algo(self.model, self.params_hm['learning_rate'], self.params_hm['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opti, milestones = [20,40,60, 80], gamma = 0.2) # reduce the learning rate to have various learning rate 
        for e in range(self.params_hm['n_epochs']):
            # go through each epoch and learn 
            epoch_n = e+1 
            for _, x in enumerate(self.train_data_loader):
                x = x.to(self.device).float() # convert to float 
                z = self.model(x) 
                loss = simple_loss_algo(z, x) 
                loss_batch = loss 
                opti.zero_grad()
                loss_batch.backward(retain_graph=True)
                opti.step() 
            scheduler.step() 
            if (e+1)%1 ==0:
                epoch_loss = loss_batch.item() 
                print(f'epoch:{epoch_n}')
                print(f'learning_rate:{scheduler.get_lr()[0]}')
                print(f'loss value:{epoch_loss}') 