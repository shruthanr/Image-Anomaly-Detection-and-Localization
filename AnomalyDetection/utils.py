import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib

def compute_distance_matrix(x, y, device='cpu'):
    x, y = x.to(device), y.to(device)
    N, D, M = x.size(0), x.size(1), y.size(0)
    x = x.unsqueeze(1).expand(N, M, D)
    y = y.unsqueeze(0).expand(N, M, D)
    dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
    return dist_matrix


def denormalize(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def concatenate_embeddings(x, y, device):
    x, y = x.to(device), y.to(device)
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z
