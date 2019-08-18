import torch
import torch.nn as nn
import torchvision.transforms as transforms


class AE_Block(nn.Module):
    def __init__(self, alpha, mode='add'):
        assert mode in [
            'add', 'mult', 'zeros'], 'Error: AE_Block mode must be "add" or "mutl" or "zeros".'
        super(AE_Block, self).__init__()
        self.alpha = alpha
        self.mask_transform = None
        self.mode = mode

    def forward(self, x, mask, save=False):
        out = torch.zeros(x.size())
        if self.mask_transform is None:
            self.mask_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((x.size()[1], x.size()[2])),
                transforms.ToTensor()
            ])
        if self.mode == 'add':
            out = self.alpha * self.mask_transform(mask)
            out = x + out
        elif self.mode == 'mult':
            out = self.alpha * self.mask_transform(mask) * x
        return x+out
