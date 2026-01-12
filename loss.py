import torch
from nnunetv2.training.loss.compound_losses.compound_lossses import DC_and_CE_loss
from nnunetv2.training.loss.compound_losses.dice import SoftDiceLoss

class region_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                        dice_class = SoftDiceLoss):
            
        super().__init__(soft_dice_kwargs, ce_kwargs, weight_ce, weight_dice, ignore_label,
                        dice_class)
        self.click_map = None

    def forward(self,):
        pass

    



