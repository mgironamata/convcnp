import os
#%cd convcnp
os.getcwd()

import numpy as np
import matplotlib.pyplot as plt
import time, datetime

import pandas as pd

import torch
import torch.nn as nn
#import stheno.torch as stheno

#import convcnp.data
import convcnp.data_hydro_2

from convcnp.experiment import report_loss, RunningAverage
from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple
from convcnp.architectures import SimpleConv, UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()   

start_time = time.time()
filepath = r'../data/camels_processed/maurer_model_output_complete.csv'
df = pd.read_csv(filepath)
elapsed = time.time() - start_time
print("CSV imported -- time: %.3f" %  (elapsed))

s_year = 2000
s_month = 6
s_day = 1
e_year = 2000
e_month = 7
e_day = 30

min_train_points = 10
min_test_points = 10
max_train_points = 20
max_test_points = 20

#channel_names = ['OBS_RUN','PRCP']

#s_date = str(s_year)  + '-' + str(s_month).zfill(2) + '-' + str(s_day).zfill(2)
#e_date = str(e_year)  + '-' + str(e_month).zfill(2) + '-' + str(e_day).zfill(2)
#df = df.iloc[s_date:e_date,:]

df = df[(df['YR'] >= s_year) & (df['MNTH'] >= s_month) & (df['YR'] <= e_year) & (df['MNTH'] <= e_month) & (df['hru02'] == 'hru_01')]

#kernel = stheno.Matern52().stretch(0.25)
#gen = convcnp.data.GPGenerator(kernel=kernel)

gen = convcnp.data_hydro_2.HydroGenerator(s_year=s_year,
                                            s_month=s_month,
                                            s_day=s_day,
                                            e_year=e_year,
                                            e_month=e_month,
                                            e_day=e_day,
                                            dataframe=df,
                                            batch_size = 16,
                                            num_tasks = 256,
                                            channels_c = ['OBS_RUN'],
                                            channels_t = ['OBS_RUN'],
                                            min_train_points=min_train_points,
                                            min_test_points=min_test_points,
                                            max_train_points=max_train_points,
                                            max_test_points=max_test_points)

#x_test = np.linspace(2000*365 + 30*6 + 1, 2000*365 + 30*6 + 30, 300)
x_test = np.vstack([np.linspace(0,1,30)])
#x_test = np.linspace(-2,2,300)
#gp = stheno.GP(kernel)

def plot_task(task, idx, legend):
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx][:,0])
    x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx][:,0])#
      
    # Plot context and target sets.
    plt.scatter(x_context, y_context, label='Context Set (rain)', color='black', marker='x')
    #plt.scatter(x_context, y_context[:,1], label='Context Set (flow)', color='black', marker='o')
    plt.scatter(x_target, y_target, label = 'Target Set (rain)', color='blue', marker='x')
    #plt.scatter(x_target, y_target, label = 'Target Set (flow)', color='blue', marker='o')
    #obs_x = (df['idx'][df['idx'].isin(np.arange(s_ind,e_ind+1,1))]-s_ind)/(e_ind-s_ind)
    #obs_y = df['OBS_RUN'][df['idx'].isin(np.arange(s_ind,e_ind+1,1))]
    #plt.plot(obs_x,obs_y)
    
    # Infer GP posterior.
    #post = gp  | (x_context, y_context)
    
    # Make and plot predictions on desired range.
    #gp_mean, gp_lower, gp_upper = post(x_test).marginals()
    #plt.plot(x_test, gp_mean, color='tab:green', label='Oracle GP')
    #plt.plot(x_test)
    #plt.fill_between(x_test, gp_lower, gp_upper, color='tab:green', alpha=0.1)
    if legend:
        plt.legend()

task = gen.generate_task()

fig = plt.figure(figsize=(24, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plot_task(task, i, legend=i==2)
#plt.show()

def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape (batch, n, 1).
        y (tensor): Inputs of shape (batch, m, 1).

    Returns:
        tensor: Pair-wise distances of shape (batch, n, m).
    """
    
    return (x - y.permute(0, 2, 1)) ** 2


def compute_dists_2D(x_context, x_target):
        '''
        Compute dists for psi for 2D
        '''
        
        t1 = (x_context[:, :, 0:1] - x_target.permute(0, 2, 1)[:, 0:1, :])**2
        t2 = (x_context[:, :, 1:2] - x_target.permute(0, 2, 1)[:, 1:2, :])**2
        
        return (t1 + t2)

class ConvDeepSet(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, in_channels, out_channels, init_length_scale):
        super(ConvDeepSet, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels + 1
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels), requires_grad=True)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model
    
    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between x and t.

        Returns:
            tensor: Evaluation of psi(x, t) with psi an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.lshape[1]
        
        #pdb.set_trace()

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        
        #pdb.set_tracdb.set_trace()
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Compute the extra density channel.
        # Shape: (batch, n_in, 1).
        density = torch.ones(batch_size, n_in, 1).to(device)

        # Concatenate the channel.
        # Shape: (batch, n_in, in_channels + 1).
        y_out = torch.cat([density, y], dim=2)

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels + 1).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels + 1).
        y_out = y_out.sum(1)

        # Use density channel to normalize convolution.
        density, conv = y_out[..., :1], y_out[..., 1:]
        normalized_conv = conv / (density + 1e-8)
        y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out


class FinalLayer(nn.Module):
    """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, in_channels, init_length_scale):
        super(FinalLayer, self).__init__()
        self.out_channels = 1
        self.in_channels = in_channels
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) * torch.ones(self.in_channels), requires_grad=True)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model
    
    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between x and t.

        Returns:
            tensor: Evaluation of psi(x, t) with psi an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        y_out = y_out.sum(1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out


class ConvCNP(nn.Module):
    """One-dimensional ConvCNP model.

    Args:
        learn_length_scale (bool): Learn the length scale.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    def __init__(self, in_channels, rho, points_per_unit):
        super(ConvCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.rho = rho
        self.multiplier = 2 ** self.rho.num_halving_layers
        self.in_channels = in_channels

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit
        
        # Instantiate encoder
        self.encoder = ConvDeepSet(in_channels, out_channels=self.rho.in_channels,
                                   init_length_scale=init_length_scale)
        
        # Instantiate mean and standard deviation layers
        self.mean_layer = FinalLayer(in_channels=self.rho.out_channels,
                                     init_length_scale=init_length_scale)
        self.sigma_layer = FinalLayer(in_channels=self.rho.out_channels,
                                      init_length_scale=init_length_scale)

    def forward(self, x, y, x_out):
        """Run the model forward.

        Args:
            x (tensor): Observation locations of shape (batch, data, features).
            y (tensor): Observation values of shape (batch, data, outputs).
            x_out (tensor): Locations of outputs of shape (batch, data, features).
            
        Returns:
            tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
        """
        # Determine the grid on which to evaluate functional representation.
        x_min = min(torch.min(x[:,:,0]).cpu().numpy(),
                    torch.min(x_out[:,:,0]).cpu().numpy(), 0.) - 0.1
        x_max = max(torch.max(x[:,:,0]).cpu().numpy(),
                    torch.max(x_out[:,:,0]).cpu().numpy(), 1.) + 0.1
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multiplier))
        
        """x_min_1 = min(torch.min(x[:,:,1]).cpu().numpy(),
                    torch.min(x_out[:,:,1]).cpu().numpy(), 0.) - 0.1
        x_max_1 = max(torch.max(x[:,:,1]).cpu().numpy(),
                    torch.max(x_out[:,:,1]).cpu().numpy(), 1.) + 0.1
        num_points_1 = int(to_multiple(self.points_per_unit * (x_max_1 - x_min_1),
                                     self.multiplier))"""

        x_grid = torch.linspace(x_min, x_max, num_points).to(device)
        x_grid = x_grid[None, :, None].repeat(x.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.encoder(x, y, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], num_points)
        h = self.rho(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations.
        mean = self.mean_layer(x_grid, h, x_out)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))

        return mean, sigma

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
    

model = ConvCNP(in_channels = len(channel_names), rho=UNet(), points_per_unit=64)
model.to(device)

def train(data, model, opt):
    """Perform a training epoch."""
    ravg = RunningAverage()
    model.train()
    for step, task in enumerate(data):
        y_mean, y_std = model(task['x_context'], task['y_context'], task['x_target'])
        obj = -gaussian_logpdf(task['y_target'], y_mean, y_std, 'batched_mean')
        obj.backward()
        opt.step()
        opt.zero_grad()
        ravg.update(obj.item() / data.batch_size, data.batch_size)
    return ravg.avg


# Create a fixed set of outputs to predict at when plotting.
x_test = torch.linspace(0., 1.,100)[None, :, None].to(device)


def plot_model_task(model, task, idx, legend):
    num_functions = task['x_context'].shape[0]
    
    # Make predictions with the model.
    model.eval()
    with torch.no_grad():
        y_mean, y_std = model(task['x_context'], task['y_context'], x_test.repeat(num_functions, 1, 1))
    
    # Plot the task and the model predictions.
    x_context, y_context = to_numpy(task['x_context'][idx]), to_numpy(task['y_context'][idx][:,0])
    x_target, y_target = to_numpy(task['x_target'][idx]), to_numpy(task['y_target'][idx][:,0])
    y_mean, y_std = to_numpy(y_mean[idx][:,0]), to_numpy(y_std[idx][:,0])
    
    # Plot context and target sets.
    plt.scatter(x_context, y_context, label='Context Set', color='black')
    plt.scatter(x_target, y_target, label='Target Set', color='red')
    
    # Plot model predictions.
    plt.plot(to_numpy(x_test[0,:,0]), y_mean, label='Model Output', color='blue')
    plt.fill_between(to_numpy(x_test[0,:,0]),
                     y_mean + 2 * y_std,
                     y_mean - 2 * y_std,
                     color='tab:blue', alpha=0.2)
    if legend:
        plt.legend()

# Some training hyper-parameters:
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
PLOT_FREQ = 10

# Initialize optimizer
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Run the training loop.
for epoch in range(NUM_EPOCHS):

    # Compute training objective.
    start_time = time.time()
    train_obj = train(gen, model, opt)
    elapsed = time.time() - start_time
    
    # Plot model behaviour every now and again.
    #if epoch == 0:
    #    print('Epoch %s: NLL %.3f -- time: %.3f' % (epoch, train_obj, elapsed))
    if epoch % PLOT_FREQ == 0:
        print('Epoch %s: NLL %.3f -- time: %.3f' % (epoch, train_obj, elapsed))
        task = gen.generate_task()
        fig = plt.figure(figsize=(24, 5))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plot_model_task(model, task, idx=i, legend=i==2)
        #plt.show()

        PATH = r'C:\Users\marcg\Google Drive\MResProject\saved_models\1D_19900601_20000730_epoch%s.pt' % (str(epoch).zfill(2))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            #'loss': loss,
            }, PATH)

    else:
        print('Epoch %s: NLL %.3f -- time: %.3f' % (epoch, train_obj, elapsed))

PATH = r'C:\Users\marcg\Google Drive\MResProject\saved_models\1D_19900601_20000730.pt'
torch.save(model.state_dict(),PATH)

# Instantiate data generator for testing.
NUM_TEST_TASKS = 2048
gen_test = convcnp.data_hydro_2.HydroGenerator(s_year=s_year,
                                            s_month=s_month,
                                            s_day=s_day,
                                            e_year=e_year,
                                            e_month=e_month,
                                            e_day=e_day,
                                            dataframe=df,
                                            min_train_points = min_train_points,
                                            min_test_points = min_test_points,
                                            max_train_points = max_train_points,
                                            max_test_points = max_test_points,
                                            num_tasks=NUM_TEST_TASKS)

# Compute average task log-likelihood.
ravg = RunningAverage()
model.eval()
with torch.no_grad():
    for step, task in enumerate(gen_test):
        y_mean, y_std = model(task['x_context'], task['y_context'], task['x_target'])
        obj = -gaussian_logpdf(task['y_target'], y_mean, y_std, 'batched_mean')
        ravg.update(obj.item() / gen_test.batch_size, gen_test.batch_size)

print('Model averages a log likelihood of %.2f on unseen tasks.' % -ravg.avg)