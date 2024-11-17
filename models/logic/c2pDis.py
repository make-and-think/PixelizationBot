from .basic_layer import *
import math
from torch.nn import Parameter
#from pytorch_metric_learning import losses

'''
Margin code is borrowed from https://github.com/MuggleWang/CosFace_pytorch and https://github.com/wujiyang/Face_Pytorch.
'''
def cosine_sim(x1, x2, dim=1, eps=1e-8):
    # Compute cosine similarity between two tensors
    ip = torch.mm(x1, x2.t())  # Inner product
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1, w2).clamp(min=eps)

class MarginCosineProduct(nn.Module):
    r"""Implementation of large margin cosine distance.
    
    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        s: Norm of input feature.
        m: Margin.
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))  # Weight initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # Compute cosine similarity
        cosine = cosine_sim(input, self.weight)  # Shape: (1, 512) and (7, 512)
        
        # Convert label to one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        # Calculate output with margin
        output = self.s * (cosine - one_hot * self.m)
        torch.cuda.empty_cache()  # Очистка памяти после генерации
        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, s={self.s}, m={self.m})"

class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Ensure the function cos(theta+m) is monotonic decreasing for theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # Compute cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # Compute cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply easy margin if specified
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # Convert label to one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        # Calculate output with margins
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        torch.cuda.empty_cache()  # Очистка памяти после генерации
        return output


class MultiMarginProduct(nn.Module):
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m1=0.20, m2=0.35, easy_margin=False):
        super(MultiMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m1 = math.cos(m1)
        self.sin_m1 = math.sin(m1)

        # Ensure the function cos(theta+m) is monotonic decreasing for theta in [0°,180°]
        self.th = math.cos(math.pi - m1)
        self.mm = math.sin(math.pi - m1) * m1

    def forward(self, x, label):
        # Compute cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # Compute cos(theta + m1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m1 - sine * self.sin_m1

        # Apply easy margin if specified
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # Convert label to one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        # Calculate output with additive angular and cosine margins
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # Additive angular margin
        output = output - one_hot * self.m2  # Additive cosine margin
        output = output * self.s
        torch.cuda.empty_cache()  # Очистка памяти после генерации
        return output


class CPDis(nn.Module):
    """PatchGAN Discriminator."""
    
    def __init__(self, image_size=256, conv_dim=64, repeat_num=3, norm='SN'):
        super(CPDis, self).__init__()

        layers = []
        # Initialize the first convolutional layer
        layers.append(self._get_conv_layer(3, conv_dim, norm))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(self._get_conv_layer(curr_dim, curr_dim * 2, norm))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim *= 2

        # Final convolutional layer
        layers.append(self._get_conv_layer(curr_dim, curr_dim * 2, norm, stride=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim *= 2

        self.main = nn.Sequential(*layers)
        self.conv1 = self._get_conv_layer(curr_dim, 1, norm, bias=False)

    def _get_conv_layer(self, in_channels, out_channels, norm, stride=2, padding=1, kernel_size=4, bias=True):
        """Helper function to create a convolutional layer with optional spectral normalization."""
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        return spectral_norm(conv_layer) if norm == 'SN' else conv_layer

    def forward(self, x):
        if x.ndim == 5:
            x = x.squeeze(0)
        assert x.ndim == 4, x.ndim
        h = self.main(x)
        out_makeup = self.conv1(h)
        torch.cuda.empty_cache()  # Очистка памяти после генерации
        return out_makeup


class CPDis_cls(nn.Module):
    """PatchGAN Discriminator with Classification."""
    
    def __init__(self, image_size=256, conv_dim=64, repeat_num=3, norm='SN'):
        super(CPDis_cls, self).__init__()

        layers = []
        # Initialize the first convolutional layer
        layers.append(self._get_conv_layer(3, conv_dim, norm))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(self._get_conv_layer(curr_dim, curr_dim * 2, norm))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim *= 2

        # Final convolutional layer
        layers.append(self._get_conv_layer(curr_dim, curr_dim * 2, norm, stride=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))
        curr_dim *= 2

        self.main = nn.Sequential(*layers)
        self.conv1 = self._get_conv_layer(curr_dim, 1, norm, bias=False)
        self.classifier_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_conv = nn.Conv2d(512, 512, 1, 1, 0)
        self.classifier = MarginCosineProduct(512, 7)  # Using Large Margin Cosine Loss
        print("Using Large Margin Cosine Loss.")

    def _get_conv_layer(self, in_channels, out_channels, norm, stride=2, padding=1, kernel_size=4, bias=True):
        """Helper function to create a convolutional layer with optional spectral normalization."""
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        return spectral_norm(conv_layer) if norm == 'SN' else conv_layer

    def forward(self, x, label):
        if x.ndim == 5:
            x = x.squeeze(0)
        assert x.ndim == 4, x.ndim
        h = self.main(x)  # Shape: ([1, 512, 31, 31])
        out_cls = self.classifier_pool(h)
        out_cls = self.classifier_conv(out_cls)
        out_cls = out_cls.squeeze(-1).squeeze(-1)  # Remove last two dimensions
        out_cls = self.classifier(out_cls, label)
        out_makeup = self.conv1(h)  # Shape: ([1, 1, 30, 30])
        torch.cuda.empty_cache()  # Очистка памяти после генерации
        return out_makeup, out_cls

class SpectralNorm(object):
    def __init__(self):
        self.name = "weight"
        self.power_iterations = 1

    def compute_weight(self, module):
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        w = getattr(module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # Calculate the spectral norm
        sigma = u.dot(w.view(height, -1).mv(v))
        return w / sigma.expand_as(w)

    @staticmethod
    def apply(module):
        name = "weight"
        fn = SpectralNorm()

        try:
            # Check if parameters already exist
            u = getattr(module, name + "_u")
            v = getattr(module, name + "_v")
            w = getattr(module, name + "_bar")
        except AttributeError:
            # Initialize parameters if they do not exist
            w = getattr(module, name)
            height = w.data.shape[0]
            width = w.view(height, -1).data.shape[1]
            u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
            v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
            w_bar = Parameter(w.data)

            module.register_parameter(name + "_u", u)
            module.register_parameter(name + "_v", v)
            module.register_parameter(name + "_bar", w_bar)

        # Remove original weight from parameter list
        del module._parameters[name]

        # Compute and set the new weight
        setattr(module, name, fn.compute_weight(module))

        # Recompute weight before every forward pass
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_u']
        del module._parameters[self.name + '_v']
        del module._parameters[self.name + '_bar']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def spectral_norm(module):
    """Apply spectral normalization to a module."""
    SpectralNorm.apply(module)
    return module


def remove_spectral_norm(module):
    """Remove spectral normalization from a module."""
    name = 'weight'
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(f"spectral_norm of '{name}' not found in {module}")
