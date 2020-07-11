import matplotlib.pyplot as plt 
import numpy as np 
import torch
import scipy.stats
import scipy.special 
from .utils import *

def plot_task(task, idx, legend):
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx][:,0])
    x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx][:,0])
    y_target_val = to_numpy(task['y_target_val'][idx][:,0])
      
    # Plot context and target sets.
    plt.scatter(x_context, y_context, label='Context Set', color='yellow', marker='o')
    plt.scatter(x_target, y_target, label = 'Target Set', color='blue', marker='x')
    plt.plot(x_target, y_target_val, label='Target Mean', color='green')
    if legend:
        plt.legend()

def plot_training_loss(train_obj_list, train_nse_list):
    
    fig2 = plt.figure(figsize=(15,5))
    
    plt.subplot(1,2,1)
    plt.plot(train_obj_list,'r')
    plt.ylabel('NLL')
    plt.xlabel('# epochs')

    plt.subplot(1,2,2)
    plt.plot(train_nse_list,'b')
    plt.ylabel('NSE')
    plt.xlabel('# epochs')

    #fig2.suptitle('ConvCNP (Gaussian LL w/ new decoder)', fontsize = 16)
    plt.show()