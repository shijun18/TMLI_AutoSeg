import torch.nn as nn

from loss.dice_loss import DiceLoss
from loss.cross_entropy import CrossentropyLoss


class CEPlusDice(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A list of two tensors
        target: A list of two tensors
        other args pass to BinaryDiceLoss
    Return:
        combination loss, dice plus cross entropy
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(CEPlusDice, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        # print(predict.size())
        # print(target.size())
        assert predict.size() == target.size()
        dice = DiceLoss(weight=self.weight,ignore_index=self.ignore_index,**self.kwargs)
        dice_loss = dice(predict,target)

        ce = CrossentropyLoss(weight=self.weight)
        ce_loss = ce(predict,target)
        
        total_loss = ce_loss + dice_loss

        return total_loss



