import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F

import re
from utils import gradient

# 获取模型参数
# 参数是字典，根据key获取子字典
def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    # 匹配字符串
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    # layer1.weight -> weight
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)
# 元模型的基类，元模型可以在前向过程中传入模型参数
class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """
    # 获取模块中所有元参数的名称和参数对
    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule) else [],
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem
    # 仅返回所有元参数而不包含名称
    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

# Sequential module that can handle MetaModules
class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module '
                    '(inheriting from `nn.Module`), or a `MetaModule`. '
                    'Got type: `{0}`'.format(type(module)))
        return input


# 可处理batch形状参数
# input输入形状为 [batch_size, ..., in_features]
# params参数形状为  [batch_size, out_features,in_features]
# 输出形状为 [batch_size, ..., out_features]

# 或者正常的线性层
# 输入形状为 [..., in_features]，参数形状为 [out_features,in_features]
# 输出形状为 [..., out_features]
class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']
        # 其余维度不变，交换最后两个维度
        # weight的形状为 [batchsize, out_features,in_features]
        # 交换维度后的形状为 [batchsize, in_features,out_features]
        # input的形状为 [batchsize,...,in_features]
        # 得到的输出形状为 [batchsize,..., out_features]
        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        # 倒数第二个维度添加偏置，正确广播运算
        # bias的形状为 [batchsize, out_features]
        # unsqueeze后 为 [batchsize, 1,out_features]
        output += bias.unsqueeze(-2)
        return output

# 三角激活函数
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        # outermost_linear: If True, the last layer is linear, otherwise it is non-linear.
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        # first_layer
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # key 'net' 对应于self.net
        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations


class SimpleMLPNet(MetaModule):


    def __init__(self,
                 out_features=1,
                 hidden_features=256,
                 num_hidden_layers=3,
                 use_pe=True,
                 num_frequencies=8,
                 image_resolution=None,
                 ):
        super().__init__()

        self.use_pe = use_pe
        self.positional_encoding = PosEncodingNeRF2D(num_frequencies=num_frequencies,sidelength=image_resolution)
        if self.use_pe:
            in_features = self.positional_encoding.out_dim
        else:
            in_features = 2

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity='sine')
        #print(self)

    def forward(self, coords, params=None,return_grad=False):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = coords.clone().detach().requires_grad_(True)
        coords = coords_org

        # various input processing methods for different applications
        if self.use_pe:
            coords = self.positional_encoding(coords)

        output = self.net(coords, get_subdict(params, 'net'))

        if return_grad:
            dy_dx = gradient(output, coords_org, grad_outputs=torch.ones_like(output))
            return {'model_in': coords_org, 'model_out': output, 'dy_dx': dy_dx}
        else:
            return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, coords):
        '''Returns not only model output, but also intermediate activations.'''
        coords = coords.clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}



class PosEncodingNeRF2D(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''

    def __init__(self, num_frequencies=8,sidelength=None, use_nyquist=False):
        """

        :param num_frequencies:
        :param sidelength: for nyquist compute max number of frequencies image size
        :param use_nyquist:
        """
        super().__init__()

        self.in_features = 2  # x and y coordinates

        self.num_frequencies = num_frequencies
        if use_nyquist:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))

        self.out_dim = self.in_features + 2 * self.in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        # samples/4
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    #  sin(2^i * π * coord) 和 cos(2^i * π * coord)
    #  2 + 2*2*num_frequencies
    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), dim=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

    def get_output_dim(self):
        """Returns the output dimension of the positional encoding."""
        return self.out_dim



class ConvImgEncoder(nn.Module):
    def __init__(self, channel, image_resolution):
        super().__init__()

        if isinstance(image_resolution, tuple) or isinstance(image_resolution, list):
            image_size= image_resolution[0] * image_resolution[1]
        else:
            image_size = image_resolution ** 2

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 256, 1, 1, 0)
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.image_resolution = image_resolution

        # 输出
        self.fc=nn.Linear(image_size, 1)
        #self.fc = nn.Linear(1024, 1)


    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        # B,256,H,W
        o = self.cnn(o)

        # B,256,H*W ->B,256,1 -> B,256
        o = self.fc(self.relu_2(o).view(o.shape[0], 256, -1)).squeeze(-1)
        return o



class Conv2dResBlock(nn.Module):
    '''Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py'''
    def __init__(self, in_channel, out_channel=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            nn.ReLU()
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output


def channel_last(x):
    return x.transpose(1, 2).transpose(2, 3)



########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


###################
# Complex operators
def compl_conj(x):
    y = x.clone()
    y[..., 1::2] = -1 * y[..., 1::2]
    return y


def compl_div(x, y):
    ''' x / y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c ** 2 + d ** 2)
    outi = (b * c - a * d) / (c ** 2 + d ** 2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


def compl_mul(x, y):
    '''  x * y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out


if __name__ == '__main__':
    # test_linear=BatchLinear(in_features=2, out_features=1)
    # test_input=torch.randn(5,4,2)
    # out=test_linear(test_input)
    #
    # linear=nn.Linear(in_features=2, out_features=1)
    # out2=linear(test_input)
    simple_mlp=SimpleMLPNet(out_features=1,
                            hidden_features=64,
                            num_hidden_layers=3,
                            num_frequencies=8,
                            image_resolution=(16, 16))

    # model=simple_mlp
    # param_size=0
    # for param in model.parameters():
    #     param_size+=param.numel()*param.element_size()  # 计算参数的字节大小
    # size_res=param_size/1024  # 返回参数的字节大小

    batch_linear= BatchLinear(in_features=3, out_features=2)
    print(batch_linear)
    print(OrderedDict(batch_linear.named_parameters()))

    test_input=torch.randn(5,4,3)
    weight_params=nn.Parameter(torch.randn(5, 3, 2))
    test_params={
        'weight':  torch.randn(5, 2, 3),
        'bias': torch.randn(5, 2)
    }

    out=batch_linear(test_input)
    out2=batch_linear(test_input,test_params)

    fcb_block= FCBlock(in_features=2, out_features=1, num_hidden_layers=3,hidden_features=4)
    print(fcb_block)
    print(OrderedDict(fcb_block.named_parameters()))
    pass