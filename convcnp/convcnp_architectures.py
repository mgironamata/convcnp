import numpy as np
import torch
import torch.nn as nn

from convcnp.utils import (
    init_sequential_weights,
    to_multiple,
)

__all__ = ['DeepSet','FeatureEmbedding','ConvDeepSet','FinalLayer','ConvCNP']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureEmbedding(nn.Module):
    """Feature set embedding layer 
    
    Args:
    
    """

    def __init__(self,in_channels,out_channels):
        super(FeatureEmbedding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.f = self.build_weight_model()

    def build_weight_model(self):
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels,self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels,self.out_channels)
        )
        init_sequential_weights(model)
        return model

    def forward(self, y, a, b):
        return self.f(y)


class DeepSet(nn.Module):
    
    """Feature set embedding layer

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """
    
    def __init__(self,in_channels,dim_g):
        super(DeepSet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = dim_g
        self.g = self.build_weight_model_g()
        self.phi = self.build_weight_model_phi()

    def build_weight_model_phi(self):
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def build_weight_model_g(self):
        model = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model
    
    def forward(self, y, f, m):
        # Compute shapes.
        batch_size = y.shape[0]
        n_in = y.shape[1]
        embedding_channels = y.shape[2]

        # Shape: (batch, n_in, embedding_channels, out_channels).
        y_out = y.view(batch_size * n_in * embedding_channels, -1)
        f = f.view(batch_size * n_in * embedding_channels, -1)
        
        if self.in_channels == 2:
            y_out = torch.cat([y_out,f],dim=1)
        
        y_out = self.phi(y_out)
        y_out = y_out.view(batch_size, n_in, embedding_channels, self.out_channels)
 
        # Mask tensor
        m = m[:, :, :, None].repeat(1, 1, 1, self.out_channels)

        # Sum over the embedding_channels.
        # Shape of y_out and m: (batch, n_in, embedding_channels, out_channels).
        y_out = y_out * m
        y_out = torch.div(y_out.sum(2),m.sum(2))
        #replace empty sets with 0 values
        
        #print("Warning: empty set occurred") if y_out[y_out != y_out].shape[0] > 0 else pass
        
        y_out[y_out != y_out] = 0
        
        # Shape: (batch, n_in, out_channels).
        y_out = y_out.view(batch_size * n_in, self.out_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_in, self.out_channels)

        return y_out

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
        n_out = t.shape[1]
        
        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        
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

    def __init__(self, in_channels, rho, points_per_unit, dynamic_feature_embedding=True,static_feature_embedding=True,dynamic_embedding_dims=5,static_embedding_dims=5,static_embedding_location="after_encoder",dynamic_feature_missing_data=True,static_feature_missing_data=True,static_embedding_in_channels=2):
        super(ConvCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.rho = rho
        self.multiplier = 2 ** self.rho.num_halving_layers

        self.dynamic_feature_embedding = dynamic_feature_embedding
        self.dynamic_embedding_dims = dynamic_embedding_dims
        self.dynamic_feature_missing_data = dynamic_feature_missing_data

        self.static_embedding_in_channels = static_embedding_in_channels
        self.static_embedding_dims = static_embedding_dims
        self.static_feature_embedding = static_feature_embedding
        self.static_feature_missing_data = static_feature_missing_data
        self.static_embedding_location = static_embedding_location
        
        if self.dynamic_feature_embedding:
            self.in_channels = dynamic_embedding_dims
            self.pre_embedding_channels = 2
        else:
            self.in_channels = in_channels
            self.pre_embedding_channels = 1
        
        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit
        
        # Instantiate dynamic feature embedder
        self.dynamic_embedder = DeepSet(self.pre_embedding_channels,self.in_channels)
        
        # Instantiate static feature embedder
        if static_feature_missing_data:
            self.static_embedder = DeepSet(self.static_embedding_in_channels,self.static_embedding_dims)
        else:
            self.static_embedder = FeatureEmbedding(self.static_embedding_in_channels,self.static_embedding_dims)
            
        # Instantiate encoder
        if self.static_embedding_location == "after_encoder":
            encoder_out_channels = self.rho.in_channels-self.static_embedding_dims 
        else:
            encoder_out_channels = self.rho.in_channels
            
        self.encoder = ConvDeepSet(in_channels=self.in_channels, out_channels=encoder_out_channels,
                                   init_length_scale=init_length_scale)
                 
        # Instantiate mean and standard deviation layers
        if self.static_embedding_location == "after_rho":
            final_layer_in_channels = self.rho.out_channels+self.static_embedding_dims
        else:
            final_layer_in_channels = self.rho.out_channels
            
        self.mean_layer = FinalLayer(in_channels=final_layer_in_channels,
                                     init_length_scale=init_length_scale)
        self.sigma_layer = FinalLayer(in_channels=final_layer_in_channels,
                                      init_length_scale=init_length_scale)

    def forward(self, x, y, x_out, y_att=None, f=None, m=None, embedding=False, f_s=None, m_s=None):
        """Run the model forward.

        Args:
            x (tensor): Observation locations of shape (batch, data, features).
            y (tensor): Observation values of shape (batch, data, outputs).
            x_out (tensor): Locations of outputs of shape (batch, data, features).
            
        Returns:
            tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
        """
        # Determine the grid on which to evaluate functional representation.
        x_min = min(torch.min(x).cpu().numpy(),
                    torch.min(x_out).cpu().numpy(), 0.) - 0.1 + 0.1
        x_max = max(torch.max(x).cpu().numpy(),
                    torch.max(x_out).cpu().numpy(), 1.) + 0.1 - 0.1
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multiplier))
        
        x_grid = torch.linspace(x_min, x_max, num_points).to(device)
        x_grid = x_grid[None, :, None].repeat(x.shape[0], 1, 1)
        
        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        if self.dynamic_feature_embedding:
            y = self.dynamic_embedder(y,f,m)

        h = self.activation(self.encoder(x, y, x_grid))
    
        if (y_att != None) & (self.static_embedding_location=="after_encoder"):
            m_att = torch.tensor(np.ones(y_att.shape[2]), dtype=torch.float)[None,None,:].repeat(y_att.shape[0],y_att.shape[1],1).to(device)
            f_att = torch.tensor(np.arange(y_att.shape[2])/y_att.shape[2], dtype=torch.float)[None,None,:].repeat(y_att.shape[0],y_att.shape[1],1).to(device)
            h_s = self.static_embedder(y_att,f_att,m_att)
            h_s = h_s.repeat(1,h.shape[1],1)
            h = torch.cat([h,h_s],dim=2)
        
        h = h.permute(0, 2, 1)
        
        h = h.reshape(h.shape[0], h.shape[1], num_points)
        h = self.rho(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')
        
        if (y_att != None) & (self.static_embedding_location=="after_rho"):
            m_att = torch.tensor(np.ones(y_att.shape[2]), dtype=torch.float)[None,None,:].repeat(y_att.shape[0],y_att.shape[1],1).to(device)
            f_att = torch.tensor(np.arange(y_att.shape[2])/y_att.shape[2], dtype=torch.float)[None,None,:].repeat(y_att.shape[0],y_att.shape[1],1).to(device)
            h_s = self.static_embedder(y_att,f_att,m_att)
            h_s = h_s.repeat(1,h.shape[1],1)
            h = torch.cat([h,h_s],dim=2)

        # Produce means and standard deviations.
        mean = self.sigma_fn(self.mean_layer(x_grid, h, x_out))
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))
        
        return mean, sigma

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])