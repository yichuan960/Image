import torch
import math
import numpy as np
import torch.nn as nn

#tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]] ,[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32)
height = 16
width = 16
tensor = torch.randn((3, height, width), dtype=torch.float32)

residuals = torch.linalg.vector_norm(tensor, dim=(0))

def calculate_mask(residuals):
    median_residual = torch.median(residuals)
    inlier_loss = torch.where(residuals <= median_residual, 1.0, 0.0)

    kernel = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]).unsqueeze(0).unsqueeze(0).cuda()
    has_inlier_neighbors = torch.unsqueeze(inlier_loss, 0)
    has_inlier_neighbors = torch.nn.functional.conv2d(has_inlier_neighbors, kernel, padding = "same")
    has_inlier_neighbors = torch.where(has_inlier_neighbors >= 0.5, 1.0, 0.0)

    kernel_16 = 1/(16*16) * torch.ones((1,1,16,16)).cuda()
    if has_inlier_neighbors.shape[1] % 8 != 0:
        pad_h = 8 - (has_inlier_neighbors.shape[1] % 8) + 8
    else:
        pad_h = 8

    if has_inlier_neighbors.shape[2] % 8 != 0:
        pad_w = 8 - (has_inlier_neighbors.shape[2] % 8) + 8
    else:
        pad_w = 8

    padding = (math.ceil(pad_w/2), math.floor(pad_w/2), math.ceil(pad_h/2), math.floor(pad_h/2))
    padded_weights = torch.nn.functional.pad(has_inlier_neighbors, padding, mode = "replicate").cuda()

    is_inlier_patch = torch.nn.functional.conv2d(padded_weights.unsqueeze(0), kernel_16, stride = 8)

    is_inlier_patch = torch.nn.functional.interpolate(is_inlier_patch, scale_factor = 8)
    is_inlier_patch = is_inlier_patch.squeeze()

    padding_indexing = [padding[2]-4,-(padding[3]-4), padding[0]-4,-(padding[1]-4)]

    if padding_indexing[1] == 0:
        padding_indexing[1] = has_inlier_neighbors.shape[1] + padding_indexing[0]
    if padding_indexing[3] == 0:
        padding_indexing[3] = has_inlier_neighbors.shape[2] + padding_indexing[2]

    is_inlier_patch = is_inlier_patch[ padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]

    is_inlier_patch = torch.where(is_inlier_patch >= 0.6, 1.0, 0.0)

    mask = (is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze() >= 1e-3).cuda()

    return mask


kernel_size = 16
kernel_smooth = 1/(kernel_size*kernel_size) * torch.ones((1,1,kernel_size,kernel_size)).cuda()
kernel = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]).unsqueeze(0).unsqueeze(0).cuda()
kernel_16 = 1/(16*16) * torch.ones((1,1,16,16)).cuda()
def calculate_mask_proba(residuals):
    median_residual = torch.median(residuals)
    #inlier_loss = torch.where(residuals <= median_residual, 1.0, 0.0)
    inlier_loss = residuals

    has_inlier_neighbors = torch.unsqueeze(inlier_loss, 0)
    has_inlier_neighbors = torch.nn.functional.conv2d(has_inlier_neighbors, kernel, padding = "same")
    #has_inlier_neighbors = torch.where(has_inlier_neighbors >= 0.5, 1.0, 0.0)

    
    if has_inlier_neighbors.shape[1] % 8 != 0:
        pad_h = 8 - (has_inlier_neighbors.shape[1] % 8) + 8
    else:
        pad_h = 8

    if has_inlier_neighbors.shape[2] % 8 != 0:
        pad_w = 8 - (has_inlier_neighbors.shape[2] % 8) + 8
    else:
        pad_w = 8

    padding = (math.ceil(pad_w/2), math.floor(pad_w/2), math.ceil(pad_h/2), math.floor(pad_h/2))
    padded_weights = torch.nn.functional.pad(has_inlier_neighbors, padding, mode = "replicate")

    is_inlier_patch = torch.nn.functional.conv2d(padded_weights.unsqueeze(0), kernel_16, stride = 8)

    is_inlier_patch = torch.nn.functional.interpolate(is_inlier_patch, scale_factor = 8)
    is_inlier_patch = is_inlier_patch.squeeze()

    padding_indexing = [padding[2]-4,-(padding[3]-4), padding[0]-4,-(padding[1]-4)]

    if padding_indexing[1] == 0:
        padding_indexing[1] = has_inlier_neighbors.shape[1] + padding_indexing[0]
    if padding_indexing[3] == 0:
        padding_indexing[3] = has_inlier_neighbors.shape[2] + padding_indexing[2]

    is_inlier_patch = is_inlier_patch[ padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]

    #is_inlier_patch = torch.where(is_inlier_patch >= 0.6, 1.0, 0.0)

    #mask = (is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze() >= 1e-3).cuda()
    mask = is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze()
    mask_median = torch.median(mask)
    mask = mask - mask_median
    mask = torch.sigmoid(mask)

    sampling_weights = torch.rand(mask.shape, device = 'cuda:0')
    mask = torch.gt(mask, sampling_weights).float()

    
    mask = mask.unsqueeze(0)
    mask = torch.nn.functional.conv2d(mask, kernel_smooth, padding = "same").squeeze()
    mask = torch.where(mask >= 0.6, 0.0, 1.0)

    """kernel_size = 8
    kernel = 1/(kernel_size*kernel_size) * torch.ones((1,1,kernel_size,kernel_size)).cuda()
    mask = mask.unsqueeze(0)
    mask = torch.nn.functional.conv2d(mask, kernel, padding = "same").squeeze()
    mask = torch.where(mask >= 0.7, 1, 0) """

    return mask.cuda()


class RobustLoss(torch.nn.Module):
    def __init__(self):
        super(RobustLoss, self).__init__()
        # Define a learnable parameter
        self.linear1 = torch.nn.Linear(1, 1, device = 'cuda:0')
        self.sigmoid1 = torch.nn.Sigmoid()

        self.linear2 = torch.nn.Linear(1, 1, device = 'cuda:0')
        self.sigmoid2 = torch.nn.Sigmoid()

        self.linear3= torch.nn.Linear(1, 1, device = 'cuda:0')
        self.sigmoid3 = torch.nn.Sigmoid()

        kernel_size = 16
        self.kernel_16 = 1/(kernel_size*kernel_size) * torch.ones((1,1,kernel_size,16)).cuda()
        self.kernel_3 = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]).unsqueeze(0).unsqueeze(0).cuda()

    def forward(self, residuals):
        #median_residual = torch.median(residuals)
        #inlier_loss = torch.where(residuals <= median_residual, 1.0, 1e-5)
        inlier_loss = residuals - torch.median(residuals.flatten())
        
        has_inlier_neighbors = torch.unsqueeze(inlier_loss, 0)
        has_inlier_neighbors = torch.nn.functional.conv2d(has_inlier_neighbors, self.kernel_3, padding = "same")

        has_inlier_neighbors = self.linear1(has_inlier_neighbors)

        if has_inlier_neighbors.shape[1] % 8 != 0:
            pad_h = 8 - (has_inlier_neighbors.shape[1] % 8) + 8
        else:
            pad_h = 8

        if has_inlier_neighbors.shape[2] % 8 != 0:
            pad_w = 8 - (has_inlier_neighbors.shape[2] % 8) + 8
        else:
            pad_w = 8

        padding = (math.ceil(pad_w/2), math.floor(pad_w/2), math.ceil(pad_h/2), math.floor(pad_h/2))
        padded_weights = torch.nn.functional.pad(has_inlier_neighbors, padding, mode = "replicate").cuda()

        is_inlier_patch = torch.nn.functional.conv2d(padded_weights.unsqueeze(0), self.kernel_16, stride = 8)

        is_inlier_patch = torch.nn.functional.interpolate(is_inlier_patch, scale_factor = 8)
        is_inlier_patch = is_inlier_patch.squeeze()

        padding_indexing = [padding[2]-4,-(padding[3]-4), padding[0]-4,-(padding[1]-4)]

        if padding_indexing[1] == 0:
            padding_indexing[1] = has_inlier_neighbors.shape[1] + padding_indexing[0]
        if padding_indexing[3] == 0:
            padding_indexing[3] = has_inlier_neighbors.shape[2] + padding_indexing[2]

        is_inlier_patch = is_inlier_patch[ padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]

        is_inlier_patch= self.linear2(is_inlier_patch)

        mask = (is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze()).cuda()
        mask = self.threshold(self.linear3, self.sigmoid3, mask)

        return mask
    
    def threshold(self, linear, sigmoid, x):
        shape = x.shape
        x = linear(x.flatten().unsqueeze(1))
        x = sigmoid(x)
        x = x.reshape(shape)
        return x
    
    def zero_center(self, x):
        return x - torch.mean(x)

