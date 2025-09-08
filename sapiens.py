import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F
import os
import glob
import matplotlib.pyplot as plt


from sapiens.classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE


class SapiensSet(torch.utils.data.Dataset):
    def __init__(self, folders, crop_size=[1024, 768], img_size=[1280, 940]):
        self.crop_size = crop_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
                transforms.Resize(size=crop_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])


        self.val_list = self.get_image_ls(folders)
        self.number_samples = len(self.val_list)


    def __len__(self):
        return len(self.val_list)

    
    def get_image_ls(self, folders):
        img_ls = []
        for folder in folders:
            path_image = folder['path_image']
            img_ls.extend(path_image)
        
        return img_ls

    def __getitem__(self, index):
        val_item = self.val_list[index]
        # Load training image
        image = cv2.cvtColor(cv2.imread(val_item), cv2.COLOR_BGR2RGB)

        input = self.transform(image)

        subject_id = val_item.split('/')[2]
        outfit = val_item.split('/')[3]
        take_id = val_item.split('/')[4]
        img_name = val_item.split('/')[-1]
        batch_name = '_'.join([subject_id, outfit, take_id, img_name])

        return input, batch_name


class SapiensWrapper(nn.Module):

    """
    Sapiens wrapper using huggingface transformer implementation.
    """
    def __init__(self,
                 model_path: str = '.checkpoints/Sapiens/sapiens_lite_host/torchscript/seg/checkpoints/sapiens_0.3b/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2',
                 freeze=True,
                 img_size=[1024, 768],
                 layer_num=None):
        super().__init__()
        if layer_num == None:
            if "0.3b" in model_path:
                self.layer_num = 24
            else:
                self.layer_num = 48
        else:
            self.layer_num = layer_num
        self.model = torch.jit.load(model_path)

        self.save_dir = 'tmp/sapiens'
        os.makedirs(self.save_dir, exist_ok=True)

        if freeze:
            self._freeze()
        
    def seg_save_and_viz(
        self, image, result, image_name, 
        classes=GOLIATH_CLASSES, palette=GOLIATH_PALETTE, save_vis=False, title=None, threshold=0.3,
    ):

        if save_vis:
            save_vis_path = os.path.join(self.save_dir, image_name)
            # image: [C,H,W] -> [H,W,C], float [0,1]
            image = image.permute(1, 2, 0).cpu().numpy()
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)  # 转成 uint8

            # seg_logits resize 到原图大小
            seg_logits = F.interpolate(
                result.unsqueeze(0), size=image.shape[:2], mode="bilinear", align_corners=False
            ).squeeze(0)

            if seg_logits.shape[0] > 1:
                pred_sem_seg = seg_logits.argmax(dim=0).cpu().numpy()
            else:
                seg_logits = seg_logits.sigmoid()
                pred_sem_seg = (seg_logits > threshold).cpu().numpy().astype(np.uint8)

            # palette mask
            num_classes = len(classes)
            ids = np.unique(pred_sem_seg)[::-1]
            legal_indices = ids < num_classes
            ids = ids[legal_indices]
            colors = [palette[label] for label in ids]

            mask = np.zeros_like(image_uint8)
            for label, color in zip(ids, colors):
                mask[pred_sem_seg == label] = color

            # 可视化合成
            vis_image = mask.astype(np.uint8)

            # 拼接左右
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

            # 保存
            cv2.imwrite(save_vis_path, vis_image)
        
    def forward(self, image):

        with torch.no_grad():
            results = self.model(image)

            return results

    def _freeze(self):
        # print(f"======== Freezing SapiensWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False


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



if __name__ == "__main__":
    model = SapiensWrapper()
    model.eval()


    folders = extract_files('.datasets/4ddress')
    dataset = SapiensSet(folders)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, (image, batch_name) in enumerate(dataloader):
        output = model(image) # (bs, 28, h, w)

        for bs in range(image.shape[0]):
            model.seg_save_and_viz(image[bs], output[bs], batch_name[bs], save_vis=True)

