
import torch
import torch.nn as nn


def init_device(args):
    if args["cuda"] and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        if args["cuda_deterministic"]:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
    torch.set_num_threads(args["torch_threads"])
    return device


def get_active_func(activation_func):
    activations = {
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "selu": nn.SELU,
        "hardswish": nn.Hardswish,
        "identity": nn.Identity,
    }
    assert activation_func in activations, f"activation function {activation_func} not supported"
    return activations[activation_func]()


def get_init_method(initialization_method):
    return nn.init.__dict__[initialization_method]


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a * e**2 / 2 + b * d * (abs(e) - d / 2)




def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
