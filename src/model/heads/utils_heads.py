import torch
import torch.nn as nn

import torchvision
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple

Norm = nn.LayerNorm

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, bias='auto',
                 inplace=True, affine=True):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.use_norm = norm_layer is not None
        self.use_activation = activation_layer is not None
        if bias == 'auto':
            bias = not self.use_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_norm:
            self.bn = norm_layer(out_channels, affine=affine)
        if self.use_activation:
            self.activation = activation_layer(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


def spatial_normalize_pred(pred, image, ignore_index=255):
    prob = {}
    for t in pred.keys():
        task_pred = pred[t]
        batch_size, num_classes, H, W = task_pred.size()
        # check for ignore_index in input image, arising for example from data augmentation
        ignore_mask = (nn.functional.interpolate(image, size=(
            H, W), mode='nearest') == ignore_index).any(dim=1, keepdim=True)
        # so they won't contribute to the softmax
        task_pred[ignore_mask.expand_as(task_pred)] = -float('inf')
        c_probs = nn.functional.softmax(
            task_pred.view(batch_size, num_classes, -1), dim=2)
        # if the whole input image consisted of ignore regions, then context probs should just be zero
        prob[t] = torch.where(torch.isnan(
            c_probs), torch.zeros_like(c_probs), c_probs)
    return prob





def prep_a_net(model_name, shall_pretrain):
    model = getattr(torchvision.models, model_name)(shall_pretrain)
    if "resnet" in model_name:
        model.last_layer_name = 'fc'
    elif "mobilenet_v2" in model_name:
        model.last_layer_name = 'classifier'
    return model

def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")

def random_crop(im, size, pad_size=0):
    """Performs random crop (CHW format)."""
    if pad_size > 0:
        im = zero_pad(im=im, pad_size=pad_size)
    h, w = im.shape[1:]
    if size == h:
        return im
    y = np.random.randint(0, h - size)
    x = np.random.randint(0, w - size)
    im_crop = im[:, y : (y + size), x : (x + size)]
    assert im_crop.shape[1:] == (size, size)
    return im_crop

def get_patch(images, action_sequence, patch_size):
    """Get small patch of the original image"""
    batch_size = images.size(0)
    image_size = images.size(2)

    patch_coordinate = torch.floor(action_sequence * (image_size - patch_size)).int()
    patches = []
    for i in range(batch_size):
        per_patch = images[i, :,
                    (patch_coordinate[i, 0].item()): ((patch_coordinate[i, 0] + patch_size).item()),
                    (patch_coordinate[i, 1].item()): ((patch_coordinate[i, 1] + patch_size).item())]

        patches.append(per_patch.view(1, per_patch.size(0), per_patch.size(1), per_patch.size(2)))

    return torch.cat(patches, 0)


class AnyAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False):
        super(AnyAttention, self).__init__()
        self.norm_q, self.norm_k, self.norm_v = Norm(dim), Norm(dim), Norm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = (dim / num_heads) ** (-0.5)
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, dim)

    def get_qkv(self, q, k, v, qpos, kpos):
        q = apply_pos(q, qpos, self.num_heads)
        k = apply_pos(k, kpos, self.num_heads)
        v = apply_pos(v, None, 0)
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v

    def forward(self, q=None, k=None, v=None, qpos=None, kpos=None, mask=None, rel_pos=None):
        q, k, v = self.get_qkv(q, k, v, qpos, kpos)

        # reshape
        q = rearrange(q, "b n (g c) -> b n g c", g=self.num_heads)
        k = rearrange(k, "b n (g c) -> b n g c", g=self.num_heads)
        v = rearrange(v, "b n (g c) -> b n g c", g=self.num_heads)

        # attn matrix calculation
        attn = torch.einsum("b q g c, b k g c -> b q g k", q, k)
        if rel_pos is not None:
            attn = rel_pos(q, attn)
        attn *= self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=float('-inf'))
        #attn = F.softmax(attn, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(mask.bool(), value=0)
        out = torch.einsum("b q g k, b k g c -> b q g c", attn, v.float())
        out = rearrange(out, "b q g c -> b q (g c)")
        out = self.proj(out)
        return out


def apply_pos(tensor, pos, num_heads):
    if pos is None:
        return tensor
    elif len(tensor.shape) != len(pos.shape):
        tensor = rearrange(tensor, "b n (g c) -> b n g c", g=num_heads)
        tensor = tensor + pos
        tensor = rearrange(tensor, "b n g c -> b n (g c)")
    else:
        tensor = tensor + pos
    return tensor

class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type=None,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 dilation=1,
                 pad_to_patch_size=True,
                 norm_layer=None):
        super(PatchEmbed, self).__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size
        self.pad_to_patch_size = pad_to_patch_size

        # The default setting of patch size is equal to kernel size.
        patch_size = kernel_size
        if isinstance(patch_size, int):
            patch_size = to_2tuple(patch_size)
        elif isinstance(patch_size, tuple):
            if len(patch_size) == 1:
                patch_size = to_2tuple(patch_size[0])
            assert len(patch_size) == 2, \
                f'The size of patch should have length 1 or 2, ' \
                f'but got {len(patch_size)}'

        self.patch_size = patch_size
        # Use conv layer to embed
        #self.projection = nn.Unfold(kernel_size=(14, 14), stride=(8, 8), padding=(3, 3))
        conv_type = conv_type or 'Conv2d'
        if conv_type == 'Conv2d':
            self.projection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        else:
            raise NotImplementedError
        if norm_layer is not None:
            self.norm = norm_layer(embed_dims)
        else:
            self.norm = None

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        # TODO: Process overlapping op
        if self.pad_to_patch_size:
            # Modify H, W to multiple of patch size.
            if H % self.patch_size[0] != 0:
                x = F.pad(
                    x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            if W % self.patch_size[1] != 0:
                x = F.pad(
                    x, (0, self.patch_size[1] - W % self.patch_size[1], 0, 0))
        x = self.projection(x)
        self.DH, self.DW = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.DH, w=self.DW)
        return x