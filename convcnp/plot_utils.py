import matplotlib.pyplot as plt 
import numpy as np 
import torch
import scipy.stats
import scipy.special
from typing import Dict, Tuple
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

def plot_training_loss(train_loss, test_loss):
    
    fig = plt.figure(figsize=(7,5))
    
    plt.plot(train_loss,'r',label='train NLL')
    plt.plot(test_loss, 'b', label='test NLL')
    plt.ylabel('NLL')
    plt.xlabel('# epochs')

    plt.legend()

    #fig.suptitle('ConvCNP (Gaussian LL w/ new decoder)', fontsize = 16)
    plt.show()

def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate empirical cummulative density function
    
    Parameters
    ----------
    x : np.ndarray
        Array containing the data
    
    Returns
    -------
    x : np.ndarray
        Array containing the sorted metric values
    y : np.ndarray]
        Array containing the sorted cdf values
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys