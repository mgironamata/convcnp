import torch.nn as nn

from convcnp.utils import (
    init_sequential_weights,
    init_layer_weights,
    pad_concat
)

__all__ = ['SimpleConv', 'UNet', 'SeparableConv1d', 'Conv1d', 'DepthSepConv1d']


class SimpleConv(nn.Module):
    """Small convolutional architecture from 1d experiments in the paper.
    This is a 4-layer convolutional network with fixed stride and channels,
    using ReLU activations.

    Args:
        in_channels (int, optional): Number of channels on the input to the
            network. Defaults to 8.
        out_channels (int, optional): Number of channels on the output by the
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.ReLU()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=self.out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        init_sequential_weights(self.conv_net)
        self.num_halving_layers = 0

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        return self.conv_net(x)


class UNet(nn.Module):
    """Large convolutional architecture from 1d experiments in the paper.
    This is a 12-layer residual network with skip connections implemented by
    concatenation.

    Args:
        in_channels (int, optional): Number of channels on the input to
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8):
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = 16
        self.num_halving_layers = 6

        self.l1 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l2 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l3 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l4 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l5 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l6 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=8 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l8 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l9 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=2 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l10 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=2 * self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l11 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l12 = nn.ConvTranspose1d(in_channels=2 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        return pad_concat(x, h12)

    
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """Implementation of a depthwise separable convolution for 1d signals

        Parameters
        ----------
        in_channels : int
            Number of channels of incoming signal.
        out_channels : int
            Number of channels for outgoing signal
        kernel_size : int
            Width of convolutional filter
        stride : int
            Stride of convolution

        """
        super(SeparableConv1d, self).__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   padding=0,
                                   stride=1)

    def forward(self, x):
        """"Forward pass through the depthwise separable layer

        Parameters
        ----------
        x : torch.tensor
             batch of input signals (batch x in_channels x L_in).

        Returns
        -------
        torch.tensor
            batch of output signals (batch x out_channels x L_out)

        """
        x = self.depthwise(x)
        return self.pointwise(x)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """Implementation of a 1d convolution for 1d signals. Here we compute
        the padding to maintain the spatial dimensions of the signal.

        Parameters
        ----------
        in_channels : int
            Number of channels of incoming signal.
        out_channels : int
            Number of channels for outgoing signal
        kernel_size : int
            Width of convolutional filter
        stride : int
            Stride of convolution

        """
        super(Conv1d, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)

    def forward(self, x):
        """"Forward pass through the depthwise separable layer

        Parameters
        ----------
        x : torch.tensor
             batch of input signals (batch x in_channels x L_in).

        Returns
        -------
        torch.tensor
            batch of output signals (batch x out_channels x L_out)

        """
        return self.conv(x)


class DepthSepConv1d(nn.Module):
    """CNN to use as decoder. Allows for parametrizing number of layers,
    channels, separable / not, and kernel size.

    Args:
        in_channels (int, optional): Number of channels on the input to the
            network. Defaults to 8.
        conv_channels (int, optional): Number of channels in hidden layers of
            network. Defaults to 64.
        out_channels (int, optional): Number of channels on the output by the
            network. Defaults to 8.
        num_layers (int, optional): Number of hidden layers in network. 
            Defaults to 4.
        kernel_size (int, optional): Width of convolutional kernel.
            Defaults to 7
        separable (bool, optional): Switch between depthwise separable and
            normal convolutions. Defaults to False (regular).

    """

    def __init__(self,
                 in_channels=8,
                 conv_channels=64, #MGM addition -- rename "hidden_channels to conv_channels"
                 out_channels=8,
                 num_layers=7,
                 kernel_size=15,
                 separable=True):
        super(DepthSepConv1d, self).__init__() #MGM addition -- replace "Conv1D by DepthSepConv1D" in this line
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.out_channels = out_channels  #MGM addition
        self.num_halving_layers = 0 #MGM addition

        # Switch between depthwise separable and standard convolutions
        layer = SeparableConv1d if separable else Conv1d
        self.activation = nn.ReLU()

        # Initialize operations with single hidden layer
        operations = nn.ModuleList([
            layer(in_channels=in_channels,
                  out_channels=conv_channels,
                  kernel_size=kernel_size)
        ])
        operations.append(nn.ReLU())

        # Add hidden layers as required
        for _ in range(1, num_layers - 1):
            operations.append(layer(in_channels=conv_channels,
                                    out_channels=conv_channels,
                                    kernel_size=kernel_size))
            operations.append(nn.ReLU())

        # Add final convolution layer
        operations.append(layer(in_channels=conv_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size))

        # Initialize network
        self.conv_net = nn.Sequential(*operations)
        init_sequential_weights(self.conv_net)

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        return self.conv_net(x)