import torch
import torch.nn as nn
import torchvision.transforms as transforms


class AE_Resize:
    def __init__(self, size):
        self.size = size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.size[2], self.size[3])),
            transforms.ToTensor()
        ])

    def __call__(self, tensor):
        # assert tensor.size()[0] == self.size[0], 'batch size of tensor must be "{}" but get a tensor with batch size of "{}"'.format(
        #     self.size[0], tensor.size()[0])
        out = torch.zeros(
            (tensor.size()[0], self.size[1], self.size[2], self.size[3]), device=tensor.device)
        for batch in range(tensor.size()[0]):
            transformed = self.transform(tensor[batch, :, :].cpu())
            if tensor.device.type == "cuda":
                transformed = transformed.cuda(device=tensor.device)
            out[batch, :, :, :] = transformed
        return out
