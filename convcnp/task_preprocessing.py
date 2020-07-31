import torch
import numpy as np 
from .utils import device
import pdb

__all__ = ['task_mask',
           'random_mask',
           'feature_identity',
           'prep_task',
           ]    

def task_mask(task,context_mask,target_mask,concat_static_features=False,device=device):

    batch_size = task['x_context'].shape[0]
    num_test_points = task['x_target'].shape[1]
    num_train_points = task['x_context'].shape[1]
    if concat_static_features:
        dim_s = task['y_att'].shape[2]
    else:
        dim_s = 0

    context_mask = context_mask + np.ones(dim_s).tolist()
    target_mask = target_mask + np.ones(dim_s).tolist()

    task['m_context'] = torch.tensor(context_mask, dtype=torch.float)[None,None,:].repeat(batch_size,num_train_points,1).to(device)
    task['m_target'] = torch.tensor(target_mask, dtype=torch.float)[None,None,:].repeat(batch_size,num_test_points,1).to(device)
    
    return task

def random_mask(task, dropout_rate):

    context = torch.distributions.Bernoulli(task['m_context']*(1-dropout_rate))
    target = torch.distributions.Bernoulli(task['m_target']*(1-dropout_rate))
    
    task['m_context'] = task['m_context']*context.sample()
    task['m_target'] = task['m_target']*target.sample()

    return task

def feature_identity(task,device=device):    

    batch_size = task['x_context'].shape[0]
    num_test_points = task['x_target'].shape[1]
    num_train_points = task['x_context'].shape[1]
    dim_y = task['y_context'].shape[2]
    f_labels = np.arange(dim_y)/dim_y

    task['f_context'] = torch.tensor(f_labels, dtype=torch.float)[None,None,:].repeat(batch_size,num_train_points,1).to(device)
    task['f_target'] = torch.tensor(f_labels, dtype=torch.float)[None,None,:].repeat(batch_size,num_test_points,1).to(device)

    return task

def prep_task(task,context_mask,target_mask,dropout_rate=0,embedding=False,observe_at_target=False,concat_static_features=False,device=device):
    
    if embedding == True: 
        task = task_mask(task,context_mask,target_mask,concat_static_features=concat_static_features,device=device)
        task = random_mask(task,dropout_rate)
        task = feature_identity(task,device=device)

        if observe_at_target == True:
            task['x_context'] = torch.cat([task['x_context'],task['x_target']],dim=1)
            task['y_context'] = torch.cat([task['y_context'],task['y_target']],dim=1)
            task['m'] = torch.cat([task['m_context'],task['m_target']],dim=1)
            task['feature'] = torch.cat([task['f_context'],task['f_target']],dim=1)
        
        elif observe_at_target == False:
            task['m'] = task['m_context']
            task['feature'] = task['f_context']

    elif embedding == False:
        
        if observe_at_target == True:
            task['x_context'] = torch.cat([task['x_context'],task['x_target']],dim=1)
            task['y_context'] = torch.cat([task['y_context'],task['y_target']],dim=1)

    task['y_target'] = task['y_target'][:,:,0].unsqueeze(dim=2)

    return task