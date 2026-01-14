import torch 
import pathlib 
import sys 


def weight_grad_to_0(grad):
    """freeze the weight of the first chan of a convolution"""
    grad = grad.clone()
    grad[:,0] = 0 
    return grad

def bias_grad_to_0(grad):
    """freeze the bias of the first chan of a convolution"""
    grad[0] = 0 
    return grad

def modify_network_for_click_in_training(network,n_chan_add):
    """ 
    main func that modify the network and freeze weight and bias of the encoder   
    :param network: pytorch PlainConvUnet 
    :param n_chan_add: number of channels to add in the first conv of the network 
    """
    #retrieving the first conv of the encoder
    first_layer = network.encoder.stages[0]
    input_conv = first_layer[0].convs[0].conv   
    conv_shape = input_conv.weight.shape # output,input,d,h,w
    kernel_size = input_conv.kernel_size
    stride = input_conv.stride
    padding = input_conv.padding
    #building the new conv 
    new_conv = torch.nn.Conv3d(conv_shape[1] + n_chan_add,32,kernel_size = kernel_size ,stride = stride , padding = padding) # the dimension to change (for input channels) is the second one 
    torch.nn.init.normal_(new_conv.weight,mean = 0, std = 0.001)
    torch.nn.init.constant_(new_conv.bias,0)
    with torch.no_grad():
        new_conv.weight[:,0] = input_conv.weight[:,0]
        network.encoder.stages[0][0].convs[0].conv = new_conv
    
    #freezing the encoder part for the training ! 
    for stage in network.encoder.stages[1:]: 
        for param in stage.parameters():
            param.requires_grad = False
    # stage 0 has to be done manually 
    for param in network.encoder.stages[0][0].convs[1].parameters():
        param.requires_grad = False
    
    # freezing the first channel of the first conv 
    network.encoder.stages[0][0].convs[0].conv.weight.register_hook(weight_grad_to_0)
    network.encoder.stages[0][0].convs[0].conv.bias.register_hook(bias_grad_to_0)
    
    return network