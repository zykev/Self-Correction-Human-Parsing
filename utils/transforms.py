# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch

class BRG2Tensor_transform(object):
    def __call__(self, pic):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

class BGR2RGB_transform(object):
    def __call__(self, tensor):
        return tensor[[2,1,0],:,:]

def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, input_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def transform_parsing(pred, center, scale, width, height, input_size):

    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    target_pred = cv2.warpAffine(
            pred,
            trans,
            (int(width), int(height)), #(int(width), int(height)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))

    return target_pred

import torch
import torch.nn.functional as F

def cv2_to_torch_affine(trans, in_size, out_size):
    """
    把 OpenCV 像素坐标系的 affine (2x3) 转成 PyTorch grid_sample 的归一化 affine。
    in_size: (H_in, W_in)
    out_size: (H_out, W_out)
    """
    in_h, in_w = in_size
    out_h, out_w = out_size

    # 像素 -> 归一化 [-1,1] 的变换矩阵
    norm_in = np.array([[2.0 / (in_w - 1), 0, -1],
                        [0, 2.0 / (in_h - 1), -1],
                        [0, 0, 1]], dtype=np.float32)

    # 归一化 -> 像素的逆变换
    norm_out_inv = np.array([[ (out_w - 1) / 2.0, 0, (out_w - 1) / 2.0],
                             [0, (out_h - 1) / 2.0, (out_h - 1) / 2.0],
                             [0, 0, 1]], dtype=np.float32)
    

    # 拼接成 3x3
    trans_3x3 = np.vstack([trans, [0,0,1]])  # (3,3)

    # 转换：归一化系下的 affine
    theta = norm_in @ np.linalg.inv(trans_3x3) @ norm_out_inv
    theta = theta[0:2, :]  # (2,3)

    return theta

def transform_parsing_torch(pred, center, scale, width, height, input_size):
    """
    Batch inverse affine transform for parsing maps using PyTorch.
    Optimized for the case where center/scale are the same for all batch samples.

    Args:
        pred: (B, H, W)
        center: (2,) tensor
        scale: (2,) tensor
        width: int - output width
        height: int - output height
        input_size: (H_in, W_in)

    Returns:
        target_pred: (B, H, W)
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)  # (B, 1, H, W)

    B, C, H, W = pred.shape
    device = pred.device

    if pred.dtype not in (torch.float32, torch.float64):
        pred = pred.float()

    # 获取仿射矩阵 (2x3)
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)  # (2,3)
    theta = cv2_to_torch_affine(trans, input_size, (height, width))
    theta = torch.from_numpy(theta).to(device=device, dtype=torch.float32)
    theta = theta.unsqueeze(0).repeat(B, 1, 1)  # (B, 2, 3)

    # 生成 grid 并采样
    grid = F.affine_grid(theta, size=(B, C, height, width), align_corners=True)
    target_pred = F.grid_sample(pred.float(), grid, mode='nearest', padding_mode='zeros', align_corners=True)

    if target_pred.shape[1] == 1:
        target_pred = target_pred.squeeze(1)  # (B, H, W)

    return target_pred


def transform_logits(logits, center, scale, width, height, input_size):

    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:,:,i],
            trans,
            (int(width), int(height)), #(int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits,axis=2)

    return target_logits


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[1]), int(output_size[0])),
                             flags=cv2.INTER_LINEAR)

    return dst_img
