import torch
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import SoftDiceLoss
from torch.nn.functional import cross_entropy

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
        
if __name__ == '__main__':
    gt = torch.randint(0,3(1,1,110,120,130))
    net_output0 = torch.rand((1,3,110,120,130))
