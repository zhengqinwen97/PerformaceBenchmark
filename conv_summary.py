import torch
import torch.nn as nn

from collections import OrderedDict
from info import ConvInfo


def summary(model, input_size, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(conv_summarys)
            # print(class_name)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            if 'conv2d' in class_name.lower() and 'separable' not in class_name.lower():
                conv_summarys[m_key] = ConvInfo()
                conv_summarys[m_key].name = m_key

                conv_summarys[m_key].batch_size = input_size[0]
                conv_summarys[m_key].input_channels = input[0].size()[1]
                conv_summarys[m_key].input_h = input[0].size()[2]
                conv_summarys[m_key].input_w = input[0].size()[3]

                conv_summarys[m_key].output_channels = module.weight.size()[0]
                conv_summarys[m_key].kernel_h = module.weight.size()[2]
                conv_summarys[m_key].kernel_w = module.weight.size()[3]

                conv_summarys[m_key].stride = module.stride
                conv_summarys[m_key].paddings = module.padding

                conv_summarys[m_key].output_h = output.size()[2]
                conv_summarys[m_key].output_w = output.size()[3]

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    # if isinstance(input_size, tuple):
    #     input_size = list(input_size)

    # batch_size of 2 for batchnorm
    # x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    rand_in_size = [input_size[1], input_size[2], input_size[3]]
    x = [torch.rand(2, rand_in_size[0], rand_in_size[1], rand_in_size[2]).type(dtype)]

    # create properties
    hooks = []
    conv_summarys = OrderedDict()

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return conv_summarys
