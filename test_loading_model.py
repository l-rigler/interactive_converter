import torch 
import pathlib 
import sys 
# modifying path for import model class & scripts 
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
module_path = '/home/lrigler/miniconda3/envs/nnunet_clean/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/git_repo'
sys.path.append(module_path)

path = pathlib.Path('/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/thighs/click_project/model_thighs_no_click/nnUNet_results/Dataset001/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/fold_0')
path =  pathlib.Path('/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/4_Methodologie_Traitement_Image/#8_2022_Re-Segmentation/models/thighs/click_project/model_thighs_no_click/nnUNet_results/Dataset001/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres')
checkpoint = nnUNetPredictor()
checkpoint.initialize_from_trained_model_folder(path,use_folds = (0,))
# model = torch.load(path / 'checkpoint_final.pth',weights_only = False )
# model.load_state_dict( model['network_weights'])
network = checkpoint.network
encoder = network.encoder
first_layer = network.encoder.stages[0]
input_conv = first_layer[0].convs[0].conv

# Part where we modified the first conv block to accept the input we want :

new_conv = torch.nn.Conv3d(10,32,kernel_size = (1,3,3),stride = (1,1,1), padding = (0,1,1)) # the dimension to change (for input channels) is the second one 
torch.nn.init.normal_(new_conv.weight,mean = 0, std = 0.001)
torch.nn.init.constant_(new_conv.bias,0)
with torch.no_grad():
    new_conv.weight[:,0] = input_conv.weight[:,0]
    network.encoder.stages[0][0].convs[0].conv = new_conv

# Part for freezing the encoder and the first convolution part for the first input channel :
for stage in encoder.stages[1:]: 
    for param in stage.parameters():
        param.requires_grad = False

# stage 0 has to be done manually ! 
for param in encoder.stages[0][0].convs[1].parameters():
    param.requires_grad = False

# more specialy we need to freeze the first convolution 

def put_grad_to_0(grad):
    grad = grad.clone()
    grad[:,0] = 0 
    return grad

network.encoder.stages[0][0].convs[0].conv.weight.register_hook(put_grad_to_0)

breakpoint()