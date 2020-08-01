from scipy.special import inv_boxcox
from scipy.stats import boxcox
import numpy as np
import torch

#__all__ = [standardise,normalise,log_transform,boxcox_transform,rev_standardise,rev_normalise,rev_log_transform,rev_boxcox_transform] 

" TRANSFORMATIONS"

def standardise(x,stats=False):
    mu = np.mean(x)
    sigma = np.std(x)
    if stats:
        return (x - mu)/sigma, mu, sigma
    else:
        return (x - mu)/sigma

def normalise(x,stats=False):
    x_max = np.max(x)
    x_min = np.min(x)
    if stats:
        return (x - x_min)/(x_max - x_min), x_min, x_max
    else:
        return (x - x_min)/(x_max - x_min)

def log_transform(x, e=0):
    return np.log(x+e)

def boxcox_transform(x, e=0):
    return boxcox(x+e)

"INVERSE TRANSFORMATIONS"

def rev_standardise(x_t, mu, sigma):
    return x_t*sigma + mu

def rev_normalise(x_t, min, max):
    return x_t*(max - min) + min   
    
def rev_log_transform(x, e=1e-6):
    return np.exp(x)-e  

def rev_boxcox_transform(x, ld):
    return inv_boxcox(x, ld)

'REV BOXCOX TRANSFORM USING TENSORS'

def rev_boxcox_transform_tensor(x,ld):
        if ld == 0:
            return torch.exp(x)
        else:
            return torch.exp(torch.log(ld*x+1)/ld) 

def rev_transform(x, transform="NONE", scaling="NONE", mu=None, sigma=None, min=None, max=None, e=1, ld=None, s=0):

    if scaling == "STANRDARD":
        x = rev_standardise(x, mu=mu, sigma=sigma)
    elif scaling == "NORM":
        x = rev_normalise(x, x_min=min, x_max=max)
    else:
        pass

    if transform == "LOG"
        x = rev_log_transform(x,e=e)
    if transform == "BOXCOX"
        x = rev_boxcox_transform(x, ld=ld)

    return x - s

def rev_transform_tensor(x, transform="NONE", scaling="NONE", mu=None, sigma=None, min=None, max=None, e=1, ld=None, s=0):

    if scaling == "STANRDARD":
        x = rev_standardise(x, mu=mu, sigma=sigma)
    elif scaling == "NORM":
        x = rev_normalise(x, x_min=min, x_max=max)
    else:
        pass

    if transform == "LOG"
        x = torch.exp(x) - e
    if transform == "BOXCOX"
        x = rev_boxcox_transform_tensor(x, ld=ld)

    return x - s
    
    