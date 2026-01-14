""" based on interactive Trainer just made to test a new loss """
import numpy as np
import torch
import torchvision.transforms.functional as F
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform


import matplotlib.pyplot as plt          
from time import time, sleep
from typing import Union, Tuple, List

# #doing this to import my script ... 
# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, current_dir)

from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.resample_transforms import (
    SimulateLowResolutionTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    SpatialTransform,
    MirrorTransform,
)
from batchgenerators.transforms.utility_transforms import (
    RemoveLabelTransform,
    RenameTransform,
    NumpyToTensor,
)
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    isfile,
    save_json,
    maybe_mkdir_p,
)
from torch._dynamo import OptimizedModule

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    resample_and_save,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import (
    get_patch_size,
)
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import (
    MoveSegAsOneHotToData,
    ApplyRandomBinaryOperatorTransform,
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import (
    DownsampleSegForDSTransform2,
)

from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import SoftDiceLoss

from torch.nn.functional import cross_entropy

#TO ERASE LATER -> already in loss.py 

class region_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice = 0, ignore_label=None,
                        dice_class = SoftDiceLoss):
            
        super().__init__(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label,
                        dice_class)
        self.click_map = None
        
    def forward(self,net_output,gt):
        self.click_map = self.click_map / self.click_map.mean()
        loss = cross_entropy(net_output,gt.squeeze().long(),ignore_index = self.ignore_label,reduction = 'none')
        loss = loss * self.click_map
        loss = loss.mean()
        return loss

class combined_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                        dice_class = SoftDiceLoss):
            
        super().__init__(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label,
                        dice_class)
        self.click_map = None
        self.r_loss = region_loss( soft_dice_kwargs, ce_kwargs,ignore_label = ignore_label
                        )
        
    def forward(self,net_output,gt):

        if self.click_map==None or self.click_map.sum() == 0:
            return super().forward(net_output,gt)
        
        if self.ignore_label is not None:
            is_ignored = torch.any(gt.squeeze() == self.ignore_label,axis=[2,3])
            is_seg = 1 - (is_ignored.int().unsqueeze(2).unsqueeze(2))
            self.click_map = self.click_map * is_seg
            gt_masked = torch.where(self.click_map.unsqueeze(1) > 0,gt,self.ignore_label)
            self.r_loss.click_map = self.click_map
            return 0.5 * super().forward(net_output,gt) + 0.5 * self.r_loss.forward(net_output,gt_masked)
        

from sklearn.model_selection import KFold
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.nnUNetTrainer.git_repo import my_utils as Mutils
from . import utils 
class Trainer_test(nnUNetTrainer.nnUNetTrainer):
    """custom nnUNet Trainer that train also for interactive segmentation and prediction refinement"""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
        # max_iter=5,
        # nbr_supervised=0.5,
    ):
        """Initialize the nnU-Net trainer for muscle segmentation."""
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            device,
        )
        self.max_iter = (
            5  # max_iter # maximal number of click during an iteration of training
        )
        self.nbr_supervised = (
            0.75  # nbr_upervised  # number of image that are trained with clicks
        )
        self.dataset_json = dataset_json  # info on the dataset -> maybe not useful
        
        self.enable_deep_supervision = True
        self.num_epochs = 800 #change this to change max number of epoch 
        self.N_alpha = 800

    def _build_loss(self,):
        """function to define the loss of the model and to configure the deep supervision"""

        loss=combined_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'ignore_index':self.label_manager.ignore_label}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                if self.is_ddp and not self._do_i_compile():
                    # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                    # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                    # Anywho, the simple fix is to set a very low weight to this.
                    weights[-1] = 1e-6
                else:
                    weights[-1] = 0

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                # now wrap the loss
                loss = Mutils.DeepSupervisionWrapper(loss,deep_supervision_scales,weights)
        return loss

    def build_network_architecture(self,architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        here we modify the nnunet function to adapt the network architecture in order 
        to accept click + we freeze the encoder for training
        """
        network = super().build_network_architecture(architecture_class_name,
                                           arch_init_kwargs,
                                           arch_init_kwargs_req_import,
                                           num_input_channels,
                                           num_output_channels,
                                           enable_deep_supervision)
        network = utils.modify_network_for_click_in_training(network,len(self.label_manager.label_dict))
        return network
    
    def add_guidance(self,data,target,training_mode=True,mode='global'):
        return Mutils.click_simulation_normalized(self,data,target,training_mode,click_mode=mode)
    
    def train_step(self, batch: dict) -> dict:
    
        data = batch["data"]
        target = batch["target"]
        
        #creating and adding click chans to data :
        click_chans = torch.zeros((data.shape[0],
                                   len(self.label_manager.label_dict),
                                    *data.shape[2:]))
        data = torch.concatenate((data,click_chans),dim = 1)
        data = data.to(self.device, non_blocking=True)
        
        
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        if self.current_epoch> 0 :
            with torch.no_grad():
                net_output0=self.network(data)
                self.loss.net_output0=net_output0
                data,click_map=self.add_guidance(data,target,'global')
        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with(
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            if self.current_epoch> 0 :
                self.loss.click_map = torch.sum(torch.where(click_map>0,1,0),axis=1)
            l=self.loss(output,target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}



    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            data[:,1:]=data[:,1:]*0
            net_output0=self.network(data)
            self.loss.net_output0 = net_output0
            if self.current_epoch > 0 : 
                data,click_map=self.add_guidance(data,target,training_mode=False)
                self.loss.click_map = torch.sum(torch.where(click_map>0,1,0),axis=1)
                self.loss.loss.alpha = np.exp(-5*(self.current_epoch/self.N_alpha))

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data

            
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}
