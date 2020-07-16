import numpy as np 
import torch

def nse(obs, mean_obs, sim):
    return (1 - np.sum((sim - obs)**2)/np.sum((obs - mean_obs)**2))

def nse_tensor(obs, mean_obs, sim):
    return torch.mean(1 - torch.sum((obs - sim)**2,dim=1)/torch.sum((obs - mean_obs)**2,dim=1))

def nse_tensor_batch(obs, mean_obs, sim):
    return torch.mean(1 - torch.sum((obs - sim)**2,dim=0)/torch.sum((obs - mean_obs)**2,dim=0))

def squared_dist(x, y):
    return torch.sum((x - y)**2)

def aplha_nse(obs, mean_obs, sim):
    return np.std(sim) / np.std(obs)

def beta_nsa(obs,sim):
    return (np.mean(sim) - np.mean(obs)) / np.std(obs)

