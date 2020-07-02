import abc

import numpy as np
import pandas as pd
import torch
import os
import pdb
import random
import time
import datetime
import matplotlib.pyplot as plt

from .utils import device

__all__ = ['SawtoothGenerator', 'HydroGenerator']

def _rand(val_range, *shape):
    lower, upper = val_range
    return random.sample(range(int(lower),int(upper)),*shape)
    #return lower + np.random.rand(*shape) * (upper - lower)

def _uprank(a):
    if len(a.shape) == 1:
        return a[:, None, None]
    elif len(a.shape) == 2:
        return a[:, :, None]
    elif len(a.shape) == 3:
        return a
    else:
        return ValueError(f'Incorrect rank {len(a.shape)}.')

"""def date_to_int(row):
    return int(time.mktime(datetime.datetime(year=int(row['YR']), month=int(row['MNTH']), day=int(row['DY'])).timetuple())/86400)"""


class LambdaIterator:
    """Iterator that repeatedly generates elements from a lambda.

    Args:
        generator (function): Function that generates an element.
        num_elements (int): Number of elements to generate.
    """

    def __init__(self, generator, num_elements):
        self.generator = generator
        self.num_elements = num_elements
        self.index = 0

    def __next__(self):
        self.index += 1
        if self.index <= self.num_elements:
            return self.generator()
        else:
            raise StopIteration()

    def __iter__(self):
        return self

class DataGenerator(metaclass=abc.ABCMeta):
    """Data generator for GP samples.

    Args:
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch.
            Defaults to 256.
        x_range (tuple[float], optional): Range of the inputs. Defaults to
            [-2, 2].
        max_train_points (int, optional): Number of training points. Must be at
            least 3. Defaults to 50.
        max_test_points (int, optional): Number of testing points. Must be at
            least 3. Defaults to 50.
    """

    def __init__(self,
                 batch_size=16,
                 num_tasks=256,
                 x_range=(-2, 2),
                 min_train_points = 10,
                 min_test_points = 10,
                 max_train_points=15,
                 max_test_points=15):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.min_train_points = min_train_points
        self.min_test_points = max_test_points
        self.max_train_points = max(max_train_points, 3)
        self.max_test_points = max(max_test_points, 3)

    @abc.abstractmethod
    def sample(self,x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """
    
    def generate_task(self):
        """Generate a task.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """
        
        task = {'x': [],
                'y': [],
                'x_context': [],
                'y_context': [],
                'x_target': [],
                'y_target': [],
                }
        
        # Determine number of test and train points.
        num_train_points = np.random.randint(self.min_train_points, self.max_train_points + 1)
        num_test_points = np.random.randint(self.min_test_points, self.max_test_points + 1)
        num_points = num_train_points + num_test_points
        
        randoms = np.random.randint(0,len(self.dataframe),self.batch_size)
        ids = self.dataframe['id'].iloc[randoms].tolist()
        year = self.dataframe['YR'].iloc[randoms].tolist()
        df = self.dataframe[(self.dataframe['id'].isin(ids) | (self.dataframe['id_lag'].isin(ids)))]

        for i in range(self.batch_size):
        # Sample inputs and outputs.
            ##x = _rand(self.x_range, num_points)

            #df = self.dataframe
            
            #hru08 = df['hru08'].iloc[rand_init]
            #zone = df['zone'].iloc[rand_init]

            #year = df['DOY'].iloc[randoms].tolist()
            df_s = df[((df['id']==ids[i]) | (df['id_lag']==ids[i]))]
            s_ind, e_ind = np.array([]), np.array([])

            while (s_ind.size == 0) | (e_ind.size == 0):
                rand = np.random.randint(0,len(df_s)-60)
                DOY = df_s['DOY'].iloc[rand]
                s_ind = df_s.index[(df_s['YR']==year[i]) & (df_s['DOY']==DOY)].values
                e_ind = s_ind + 60
  
            x_ind = _rand((s_ind, e_ind),num_points)

            y, y_t = self.sample(x_ind,df_s)

            x = np.divide(np.array(x_ind) - s_ind, e_ind - s_ind)

            # Determine indices for train and test set.
            inds = np.random.permutation(len(x))
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            """# Record to task.
            task['x'].append(sorted(x))
            task['y'].append(y[np.argsort(x)])
            task['x_context'].append(x[inds_train])#(np.array(x)[inds_train])
            task['y_context'].append(y[inds_train])
            task['x_target'].append(x[inds_test])#(np.array(x)[inds_test])a
            task['y_target'].append(y[inds_test])"""

            # Record to task.
            
            task['x'].append(sorted(x))
            task['x_context'].append(x[inds_train])
            task['x_target'].append(x[inds_test])

            y_aux, y_context_aux, y_target_aux = [], [], []
            
            for i in range(len(y)):
                y_aux.append(y[i][np.argsort(x)])
                y_context_aux.append(y[i][inds_train])
            
            for i in range(len(y_t)):
                y_target_aux.append(y_t[i][inds_test])
            
            task['y'].append(np.stack(y_aux,axis=1).tolist())
            task['y_context'].append(np.stack(y_context_aux,axis=1).tolist())
            task['y_target'].append(np.stack(y_target_aux,axis=1).tolist())

            #task['y'].append(y[0][np.argsort(x)])
            #task['y_context'].append(y[0][inds_train])
            #task['y_target'].append(y[0][inds_test])

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_uprank(np.stack(v, axis=0)),
                                dtype=torch.float32).to(device)
                for k, v in task.items()}

        return task

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)

class HydroGenerator(DataGenerator):
    """ Generate samples from hydrological data"""
    
    def __init__(self,
                dataframe,
                s_year  = 2000,
                s_month = 6,
                s_day = 1,
                e_year = 2000,
                e_month = 6,
                e_day = 30,
                channels_c = ['OBS_RUN'],
                channels_t = ['OBS_RUN'],
                **kw_args):     

        self.dataframe = dataframe
        self.s_year = s_year
        self.s_month = s_month
        self.s_day = s_day
        self.e_year = e_year
        self.e_month = e_month
        self.e_day = e_day
        self.channels_c = channels_c
        self.channels_t = channels_t
        DataGenerator.__init__(self,**kw_args)
    
    def sample(self,x,df):
        return np.vstack(tuple(df[key][x] for key in self.channels_c)), np.vstack(tuple(df[key][x] for key in self.channels_t)) 
    
class SawtoothGenerator(DataGenerator):

    """Generate samples from a random sawtooth.

    Further takes in keyword arguments for :class:`.data.DataGenerator`. The
    default numbers for `max_train_points` and `max_test_points` are 100.

    Args:
        freq_dist (tuple[float], optional): Lower and upper bound for the
            random frequency. Defaults to [3, 5].
        shift_dist (tuple[float], optional): Lower and upper bound for the
            random shift. Defaults to [-5, 5].
        trunc_dist (tuple[float], optional): Lower and upper bound for the
            random truncation. Defaults to [10, 20].
    """

    def __init__(self,
                 freq_dist=(3, 5),
                 shift_dist=(-5, 5),
                 trunc_dist=(10, 20),
                 max_train_points=100,
                 max_test_points=100,
                 **kw_args):
        self.freq_dist = freq_dist
        self.shift_dist = shift_dist
        self.trunc_dist = trunc_dist
        DataGenerator.__init__(self,
                               max_train_points=max_train_points,
                               max_test_points=max_test_points,
                               **kw_args)

    def sample(self, x):
        # Sample parameters of sawtooth.
        amp = 1
        freq = _rand(self.freq_dist)
        shift = _rand(self.shift_dist)
        trunc = np.random.randint(self.trunc_dist[0], self.trunc_dist[1] + 1)

        # Construct expansion.
        x = x[:, None] + shift
        k = np.arange(1, trunc + 1)[None, :]
        return 0.5 * amp - amp / np.pi * \
               np.sum((-1) ** k * np.sin(2 * np.pi * k * freq * x) / k, axis=1)