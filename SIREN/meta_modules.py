'''Modules for hypernetwork experiments, Paper Sec. 4.4
'''

import torch
from torch import nn
from collections import OrderedDict

import utils
from SIREN import modules


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, target_module):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            target_module: MetaModule. The module whose parameters are predicted. i.e. target network.
        '''
        super().__init__()

        target_parameters = target_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        # 目标网络的每个参数都需要一个对应的MLP来预测
        # 即线性层的每个weight和bias分别需要一个MLP来预测
        for name, param in target_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = modules.FCBlock(in_features=hyper_in_features,
                                 out_features=int(torch.prod(torch.tensor(param.size()))),
                                 num_hidden_layers=hyper_hidden_layers,
                                 hidden_features=hyper_hidden_features,
                                 outermost_linear=True, nonlinearity='relu')
            self.nets.append(hn)

            # 预测权重的MLP最后一层线性层的初始化
            # 这里的weight和bias指的是目标网络的weight和bias参数，而不是预测权重的MLP最后一层
            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above) shape (B, hyper_in_features)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        PS: batch params shape is (-1,) + param_shape, where -1 is the batch size.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params

    def forward_singlesample(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        PS: batch params shape is (-1,) + param_shape, where -1 is the batch size.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            params[name] = net(z).reshape(param_shape)
        return params



#  卷积作为编码器处理2D图像
class ConvolutionalNeuralProcessImplicit2DHypernet(nn.Module):
    def __init__(self, in_features, out_features, image_resolution=None,
                 target_hidden=256, target_hidden_layers=3,use_pe=False,
                 embed_dim=256,hyper_hidden_layers=1, hyper_hidden_features=256
                 ):
        super().__init__()


        self.encoder = modules.ConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        self.target_net = modules.SimpleMLPNet(out_features=out_features, hidden_features=target_hidden, num_hidden_layers=target_hidden_layers,
                                               image_resolution=image_resolution,use_pe=use_pe)
        self.hyper_net = HyperNetwork(hyper_in_features=embed_dim, hyper_hidden_layers=hyper_hidden_layers, hyper_hidden_features=hyper_hidden_features,
                                      target_module=self.target_net)
        #print(self)

    # 修改
    def forward(self, input_img,coords):

        embedding = self.encoder(input_img)

        hypo_params = self.hyper_net(embedding)

        model_output = self.target_net(coords, params=hypo_params)

        return {'coord': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False


############################
# Initialization schemes
def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)


if __name__ == '__main__':
    # model=ConvolutionalNeuralProcessImplicit2DHypernet(in_features=1,out_features=1,image_resolution=(16,16))
    # # print(model)
    # #
    # test_img=torch.randn(8,1,16,16)
    # test_coords=torch.randn(8,10,2)
    #
    # out=model(test_img,test_coords)

    # res=utils.get_parameter_nums(model)
    # mlp_params=utils.get_parameter_nums(model.target_net)
    #
    # param_size=0
    # for param in model.target_net.parameters():
    #     param_size+=param.numel()*param.element_size()  # 计算参数的字节大小
    # size_res=param_size/1024  # 返回参数的字节大小

    target_net = modules.SimpleMLPNet(out_features=1, hidden_features=64,
                                           num_hidden_layers=3,
                                           image_resolution=(16,16))
    hyper_net = HyperNetwork(hyper_in_features=256, hyper_hidden_layers=4,
                                  hyper_hidden_features=256,
                                  target_module=target_net)
    test_embedding = torch.randn(8, 256)  # 假设的嵌入向量

    # 大小为10的字典，10对应于target_net的参数数量
    # 字典键为target_net的参数名称，值为对应的参数张量
    #
    generated_params = hyper_net(test_embedding)

    target_net_params = OrderedDict(target_net.named_parameters())

    test_simple_embedding=torch.randn(1,256)
    generated_params_single = hyper_net.forward_singlesample(test_simple_embedding)

    pass