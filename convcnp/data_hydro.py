import abc

import numpy as np
import pandas as pd
import torch
import os
#import stheno
import pdb
import random
import time
import datetime

from .utils import device

__all__ = ['GPGenerator', 'SawtoothGenerator', 'HydroGenerator']


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
                 max_train_points=5,
                 max_test_points=5):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
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
                'y_target': []}
        
        # Determine number of test and train points.
        num_train_points = np.random.randint(3, self.max_train_points + 1)
        num_test_points = np.random.randint(3, self.max_test_points + 1)
        num_points = num_train_points + num_test_points

        for i in range(self.batch_size):
            # Sample inputs and outputs.
            ##x = _rand(self.x_range, num_points)
            df = self.dataframe
            rand_init = np.random.randint(0,len(df))
            #basin = df[rand_init,13]
            #zone = df[rand_init,14]
            basin = df['basin'].iloc[rand_init]
            zone = df['zone'].iloc[rand_init]
            
            #s_ind = round(time.mktime(datetime.datetime(year=self.s_year, month=self.s_month, day=self.s_day, hour = 1).timetuple())/86400) 
            #e_ind = round(time.mktime(datetime.datetime(year=self.e_year, month=self.e_month, day=self.e_day, hour = 1).timetuple())/86400)
            s_ind = df.index[(df['YR']==self.s_year) & (df['MNTH']==self.s_month) & (df['DY']==self.s_day) & (df['basin']==basin) & (df['zone']==zone)].values
            e_ind = df.index[(df['YR']==self.e_year) & (df['MNTH']==self.e_month) & (df['DY']==self.e_day) & (df['basin']==basin) & (df['zone']==zone)].values
            #s_ind = np.where((df[:,1]==self.s_year) & (df[:,2]==self.s_month) & (df[:,3]==self.s_day) & (df[:,13]==basin) & (df[:,14]==zone))[0]
            #e_ind = np.where((df[:,1]==self.e_year) & (df[:,2]==self.e_month) & (df[:,3]==self.e_day) & (df[:,13]==basin) & (df[:,14]==zone))[0]
            
            #s_ind = self.dataframe.index[self.dataframe['idx'] == s_ind].tolist()[0]
            #e_ind = self.dataframe.index[self.dataframe['idx'] == e_ind].tolist()[0]
            x_ind = _rand((s_ind, e_ind),num_points)
            #x_ind = np.arange(s_ind,e_ind,1)
            y = self.sample(x_ind)
            #x = x_ind
            x = np.divide(np.array(x_ind) - s_ind, e_ind - s_ind)

            # Determine indices for train and test set.
            inds = np.random.permutation(len(x))#(x.shape[0])
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])
            
            #pdb.set_trace()
            #print("length of x_ind: " + str(len(x_ind)))
            #print("length of x: " + str(len(x)))
            #print("length of y: " + str(len(y)))

            # Record to task.
            task['x'].append(sorted(x))
            task['y'].append(y[np.argsort(x)].tolist())
            task['x_context'].append(x[inds_train].tolist())#(np.array(x)[inds_train])
            task['y_context'].append(y[inds_train].tolist())
            task['x_target'].append(x[inds_test].tolist())#(np.array(x)[inds_test])
            task['y_target'].append(y[inds_test].tolist())

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
                #filepath = r'/content/gdrive/My Drive/MResProject/data/camels/basin_timeseries_v1p2_modelOutput_maurer/model_output_maurer/model_output/flow_timeseries/maurer/01/01013500_05_model_output.txt',
                #filepath = r'c:\\Users\\marcg\\Google Drive\\MResProject//data/camels/basin_timeseries_v1p2_modelOutput_maurer/model_output_maurer/model_output/flow_timeseries/maurer/01/01013500_05_model_output.txt',
                #filepath = r'/mnt/c/Users/marcg/Google Drive/MResProject//data/camels/basin_timeseries_v1p2_modelOutput_maurer/model_output_maurer/model_output/flow_timeseries/maurer/01/01013500_05_model_output.txt',
                dataframe,
                s_year  = 2000,
                s_month = 6,
                s_day = 1,
                e_year = 2000,
                e_month = 6,
                e_day = 30,
                **kw_args):     

        #self.filepath = filepath
        self.dataframe = dataframe
        self.s_year = s_year
        self.s_month = s_month
        self.s_day = s_day
        self.e_year = e_year
        self.e_month = e_month
        self.e_day = e_day
        DataGenerator.__init__(self,**kw_args)

    def sample(self,x):
        #df = pd.read_table(self.filepath, sep="\s+")
        #df['idx'] = df['YR']*10000 + df['MNTH']*100 + df['DY']
        #df['idx'] = df.apply(date_to_int,axis=1)
        #df['idx'] = time.mktime(datetime.datetime(year=df['YR'], month=df['MNTH'], day=df['DY']).timetuple())/86400
        #ind_start = df.index[(df['YR'] == self.s_year) & (df['MNTH'] == self.s_month) & (df['DY'] == self.s_day)].tolist()
        #ind_end = df.index[(df['YR'] == self.e_year) & (df['MNTH'] == self.e_month) & (df['DY'] == self.e_day)].tolist()
        #return np.asarray(df['OBS_RUN'][ind_start[0]:ind_end[0]].tolist())
        #return self.dataframe[x,12]
        return np.array(self.dataframe['OBS_RUN'][x])
        #return np.asarray(self.dataframe['OBS_RUN'][self.dataframe['idx'].isin(x)].tolist())
        #return np.asarray(self.dataframe['PRCP'][self.dataframe['idx'].isin self.dataframe['OBS_RUN'][self.dataframe['idx'].isin(x)].tolist())
    """
    def range(self, data, range):
        values = np.linspace(range[0],range[1],range[1]-range[0]+1)
        data['idx'] = 
        return values
    """


class GPGenerator(DataGenerator):
    """Generate samples from a GP with a given kernel.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self):#, kernel=stheno.EQ(), **kw_args):
        self.gp = 0 #stheno.GP(kernel, graph=stheno.Graph())
        DataGenerator.__init__(self, **kw_args)

    def sample(self, x):
        return np.squeeze(self.gp(x).sample())


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