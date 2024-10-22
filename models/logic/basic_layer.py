import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class ModulationConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride=1,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ModulationConvBlock, self).__init__()
        self.in_c = input_dim
        self.out_c = output_dim
        self.ksize = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.eps = 1e-8
        weight_shape = (output_dim, input_dim, kernel_size, kernel_size)
        fan_in = kernel_size * kernel_size * input_dim
        wscale = 1.0 / np.sqrt(fan_in)

        self.weight = nn.Parameter(torch.randn(*weight_shape))
        self.wscale = wscale
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.activate_scale = np.sqrt(2.0)

    def forward(self, x, code):
        batch, in_channel, height, width = x.shape
        weight = self.weight * self.wscale
        _weight = self._compute_weight(weight, code, batch)

        x = x.view(1, batch * self.in_c, height, width)
        weight = self._reshape_weight(_weight, batch)
        x = F.conv2d(x, weight=weight, bias=None, stride=self.stride, padding=self.padding, groups=batch)

        x = x.view(batch, self.out_c, height, width) + self.bias.view(1, -1, 1, 1)
        x = self.activate(x) * self.activate_scale
        return x

    def _compute_weight(self, weight, code, batch):
        _weight = weight.view(1, self.ksize, self.ksize, self.in_c, self.out_c)
        _weight = _weight * code.view(batch, 1, 1, self.in_c, 1)
        _weight_norm = torch.sqrt(torch.sum(_weight ** 2, dim=[1, 2, 3]) + self.eps)
        return _weight / _weight_norm.view(batch, 1, 1, 1, self.out_c)

    def _reshape_weight(self, _weight, batch):
        weight = _weight.permute(1, 2, 3, 0, 4).reshape(self.ksize, self.ksize, self.in_c, batch * self.out_c)
        return weight.permute(3, 2, 0, 1)


class AliasConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(AliasConvBlock, self).__init__()
        self.use_bias = True
        
        # Initialize padding
        self.pad = self._initialize_padding(pad_type, padding)

        # Initialize normalization
        self.norm = self._initialize_normalization(norm, output_dim)

        # Initialize activation
        self.activation = self._initialize_activation(activation)

        # Initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def _initialize_padding(self, pad_type, padding):
        if pad_type == 'reflect':
            return nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            return nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            return nn.ZeroPad2d(padding)
        else:
            raise ValueError(f"Unsupported padding type: {pad_type}")

    def _initialize_normalization(self, norm, output_dim):
        if norm == 'bn':
            return nn.BatchNorm2d(output_dim)
        elif norm == 'in':
            return nn.InstanceNorm2d(output_dim)
        elif norm == 'ln':
            return LayerNorm(output_dim)
        elif norm == 'adain':
            return AdaptiveInstanceNorm2d(output_dim)
        elif norm in ['none', 'sn']:
            return None
        else:
            raise ValueError(f"Unsupported normalization: {norm}")

    def _initialize_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            return nn.PReLU()
        elif activation == 'selu':
            return nn.SELU(inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'none':
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class AliasResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(AliasResBlocks, self).__init__()
        # Initialize a list to hold the residual blocks
        self.model = nn.Sequential(*[AliasResBlock(dim, norm=norm, activation=activation, pad_type=pad_type) for _ in range(num_blocks)])

    def forward(self, x):
        # Pass the input through the sequential model
        return self.model(x)
class AliasResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(AliasResBlock, self).__init__()

        # Initialize the sequential model with two convolutional blocks
        self.model = nn.Sequential(
            AliasConvBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type),
            AliasConvBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)
        )

    def forward(self, x):
        # Store the input for the residual connection
        residual = x
        # Pass the input through the model
        out = self.model(x)
        # Add the residual connection
        out += residual
        return out
##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        # Initialize a sequential model with the specified number of residual blocks
        self.model = nn.Sequential(*[ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type) for _ in range(num_blocks)])

    def forward(self, x):
        # Pass the input through the sequential model
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        # Initialize a list to hold the linear blocks
        self.model = nn.Sequential(
            linearBlock(input_dim, input_dim, norm=norm, activation=activ),
            linearBlock(input_dim, dim, norm=norm, activation=activ),
            *[linearBlock(dim, dim, norm=norm, activation=activ) for _ in range(n_blk - 2)],
            linearBlock(dim, output_dim, norm='none', activation='none')  # no output activations
        )

    def forward(self, style0, style1=None, a=0):
        # If style1 is not provided, use style0
        style1 = style0
        # Compute the output using the model
        return self.model[3](
            (1 - a) * self.model[0:3](style0.view(style0.size(0), -1)) + 
            a * self.model[0:3](style1.view(style1.size(0), -1))
        )
##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        # Initialize the model with two convolutional blocks
        self.model = nn.Sequential(
            ConvBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type),
            ConvBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)
        )

    def forward(self, x):
        # Store the input for the residual connection
        residual = x
        # Pass the input through the model
        out = self.model(x)
        # Add the residual connection
        out += residual
        return out


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(ConvBlock, self).__init__()
        self.use_bias = True
        
        # Initialize padding
        self.pad = self._initialize_padding(pad_type, padding)

        # Initialize normalization
        self.norm = self._initialize_normalization(norm, output_dim)

        # Initialize activation
        self.activation = self._initialize_activation(activation)

        # Initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def _initialize_padding(self, pad_type, padding):
        if pad_type == 'reflect':
            return nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            return nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            return nn.ZeroPad2d(padding)
        else:
            raise ValueError(f"Unsupported padding type: {pad_type}")

    def _initialize_normalization(self, norm, output_dim):
        if norm == 'bn':
            return nn.BatchNorm2d(output_dim)
        elif norm == 'in':
            return nn.InstanceNorm2d(output_dim)
        elif norm == 'ln':
            return LayerNorm(output_dim)
        elif norm == 'adain':
            return AdaptiveInstanceNorm2d(output_dim)
        elif norm in ['none', 'sn']:
            return None
        else:
            raise ValueError(f"Unsupported normalization: {norm}")

    def _initialize_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            return nn.PReLU()
        elif activation == 'selu':
            return nn.SELU(inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'none':
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class linearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(linearBlock, self).__init__()
        use_bias = True
        
        # Initialize fully connected layer with optional spectral normalization
        self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias)) if norm == 'sn' else nn.Linear(input_dim, output_dim, bias=use_bias)

        # Initialize normalization layer based on the specified type
        self.norm = self._initialize_normalization(norm, output_dim)

        # Initialize activation function based on the specified type
        self.activation = self._initialize_activation(activation)

    def _initialize_normalization(self, norm, output_dim):
        if norm == 'bn':
            return nn.BatchNorm1d(output_dim)
        elif norm == 'in':
            return nn.InstanceNorm1d(output_dim)
        elif norm == 'ln':
            return LayerNorm(output_dim)
        elif norm in ['none', 'sn']:
            return None
        else:
            raise ValueError(f"Unsupported normalization: {norm}")

    def _initialize_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            return nn.PReLU()
        elif activation == 'selu':
            return nn.SELU(inplace=True)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'none':
            return None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out
##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        
        # Initialize running mean and variance buffers
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.weight is None or self.bias is None:
            raise ValueError("Please assign weight and bias before calling AdaIN!")
        
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance normalization
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps
        )

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_features})"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        # Initialize gamma and beta for affine transformation if required
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # Calculate the shape for broadcasting
        shape = [-1] + [1] * (x.dim() - 1)

        # Compute mean and standard deviation
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        # Normalize the input
        x = (x - mean) / (std + self.eps)

        # Apply affine transformation if required
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        # Initialize parameters if they haven't been created yet
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # Calculate the spectral norm
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        # Check if the parameters have already been created
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        # Initialize u and v parameters
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        # Remove the original weight parameter
        del self.module._parameters[self.name]

        # Register new parameters
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        # Update u and v before forwarding
        self._update_u_v()
        return self.module.forward(*args)
