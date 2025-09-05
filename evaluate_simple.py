#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   evaluate.py
@Time    :   8/4/19 3:36 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import numpy as np
import torch
import glob

from torch.utils import data
from tqdm import tqdm
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import networks
from datasets.datasets import SMPLicitValSet
from utils.transforms import BGR2RGB_transform
from utils.transforms import transform_parsing_torch


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    # Network Structure
    parser.add_argument("--arch", type=str, default='resnet101')
    # Data Preference
    parser.add_argument("--data-dir", type=str, default='.datasets/4ddress')
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--input-size", type=str, default='473,473')
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-mirror", action="store_true")
    parser.add_argument("--random-scale", action="store_true")
    # Evaluation Preference
    parser.add_argument("--log-dir", type=str, default='./tmp')
    parser.add_argument("--model-restore", type=str, default='.checkpoints/exp-schp-201908261155-lip.pth')
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--save-results", action="store_true", help="whether to save the results.")
    parser.add_argument("--flip", action="store_true", help="random flip during the test.")
    parser.add_argument("--multi-scales", type=str, default='1', help="multiple scales during the test")
    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def multi_scale_testing(model, batch_input_im, crop_size=[473, 473], flip=True, multi_scales=[1]):
    flipped_idx = (15, 14, 17, 16, 19, 18)
    if len(batch_input_im.shape) > 4:
        batch_input_im = batch_input_im.squeeze()
    if len(batch_input_im.shape) == 3:
        batch_input_im = batch_input_im.unsqueeze(0)

    interp = torch.nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)
    ms_outputs = []
    for s in multi_scales:
        interp_im = torch.nn.Upsample(scale_factor=s, mode='bilinear', align_corners=True)
        scaled_im = interp_im(batch_input_im)
        parsing_output = model(scaled_im)
        parsing_output = parsing_output[0][-1] # (bs, 20, h, w)
        output = parsing_output
        if flip:
            flipped_output = parsing_output[1]
            flipped_output[14:20, :, :] = flipped_output[flipped_idx, :, :]
            output += flipped_output.flip(dims=[-1])
            output *= 0.5
        output = interp(output)
        ms_outputs.append(output)
    ms_fused_parsing_output = torch.stack(ms_outputs)
    ms_fused_parsing_output = ms_fused_parsing_output.mean(0) # (bs, 20 , h, w)
    ms_fused_parsing_output = ms_fused_parsing_output.permute(0, 2, 3, 1)  # BHWC
    parsing = torch.argmax(ms_fused_parsing_output, dim=-1)
    parsing = parsing.data
    ms_fused_parsing_output = ms_fused_parsing_output.data
    return parsing, ms_fused_parsing_output


def get_segmentation_map(folder):
    """Create the model and start the evaluation process."""
    args = get_arguments()
    multi_scales = [float(i) for i in args.multi_scales.split(',')]
    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True
    cudnn.enabled = True

    h, w = map(int, args.input_size.split(','))
    input_size = [h, w]

    model = networks.init_model(args.arch, num_classes=args.num_classes, pretrained=None)

    IMAGE_MEAN = model.mean
    IMAGE_STD = model.std
    INPUT_SPACE = model.input_space

    if INPUT_SPACE == 'BGR':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),

        ])
    if INPUT_SPACE == 'RGB':
        transform = transforms.Compose([
            transforms.ToTensor(),
            BGR2RGB_transform(),
            transforms.Normalize(mean=IMAGE_MEAN,
                                 std=IMAGE_STD),
        ])

    # Data loader
    # folder = extract_files(args.data_dir)
    lip_test_dataset = SMPLicitValSet(folder, crop_size=input_size, transform=transform, flip=args.flip)
    num_samples = len(lip_test_dataset)
    print('Totoal testing sample numbers: {}'.format(num_samples))
    testloader = data.DataLoader(lip_test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Load model weight
    state_dict = torch.load(args.model_restore)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    if args.save_results:
        sp_results_dir = os.path.join(args.log_dir, 'sp_results')
        if not os.path.exists(sp_results_dir):
            os.makedirs(sp_results_dir)

    if args.save_results:
        palette = get_palette(20)
    with torch.no_grad():
        parsing_res = []
        for idx, batch in enumerate(tqdm(testloader)):
            image, img_name = batch
            if (len(image.shape) > 4):
                image = image.squeeze()
            c = lip_test_dataset.person_center
            s = lip_test_dataset.s
            w = lip_test_dataset.w + 1
            h = lip_test_dataset.h + 1
            # scales[idx, :] = s
            # centers[idx, :] = c
            parsing, logits = multi_scale_testing(model, image.cuda(), crop_size=input_size, flip=args.flip,
                                                  multi_scales=multi_scales) # (bs, h, w) (bs, nclasses, h, w)
            parsing_result = transform_parsing_torch(parsing, c, s, w, h, input_size)
            parsing_res.append(parsing_result)
            if args.save_results:
                for bs in range(len(img_name)):
                    parsing_result_path = os.path.join(sp_results_dir, img_name[bs])
                    output_im = PILImage.fromarray(parsing_result[bs].cpu().numpy().astype(np.uint8))
                    output_im.putpalette(palette)
                    output_im.save(parsing_result_path)
        
        parsing_res = torch.vstack(parsing_res)

    # ===== 清理模型和 CUDA 缓存 =====
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return parsing_res


def extract_files(root_folder, subject_outfit= ['Inner', 'Outer'], select_view = '0004'):
    process_folders = []
    for subject_id in sorted(os.listdir(root_folder)):
        if subject_id in ['00148', '00149_1', '00149_2']:
            subject_dir = os.path.join(root_folder, subject_id)
            for outfit in subject_outfit:
                outfit_dir = os.path.join(subject_dir, outfit)
                if os.path.exists(outfit_dir):
                    take_dir_list = sorted(os.listdir(outfit_dir))
                    for take_id in take_dir_list:
                        take_dir = os.path.join(outfit_dir, take_id)
                        process_folders.append(take_dir)
                else:
                    continue

    res = []
    for process_folder in process_folders:
        # process folder is one task for one outfit in one subject
        # print('Processing folder: ', process_folder)
        path_image = os.path.join(process_folder, 'Capture/', select_view, 'images')
        path_smpl_prediction = os.path.join(process_folder, 'SMPL')
        # path_segmentation = os.path.join(process_folder, 'Capture/', select_view, 'images')
        # path_instance_segmentation = os.path.join(process_folder, 'Capture/', select_view, 'masks')


        img_files = sorted(glob.glob(os.path.join(path_image, '*.png')))
        img_files = [img_files[0]]
        # mask_files = sorted(glob.glob(os.path.join(path_instance_segmentation, '*.png')))
        smpl_files = sorted(glob.glob(os.path.join(path_smpl_prediction, '*_smpl.pkl')))
        smpl_files = [smpl_files[0]]
        # seg_files = sorted(glob.glob(os.path.join(path_segmentation, '*.png')))

        assert len(img_files) == len(smpl_files)

        res.append({
            'process_folder': process_folder,
            'camera_view': select_view,
            'path_image': img_files,
            'path_smpl': smpl_files,
        })

    return res

if __name__ == '__main__':
    folders = extract_files('.datasets/4ddress')
    get_segmentation_map(folders)
