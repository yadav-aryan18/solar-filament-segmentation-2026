import os
import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import cv2
from collections import defaultdict
from utils.preprocessing import preprocess_image
from utils.coco_utils import polygon_to_mask

class SolarDataset(Dataset):
    """
    PyTorch Dataset for Solar Filament Segmentation.
    Implements DYNAMIC patched loading for training and STATIC patched loading for validation.
    """
    def __init__(self, json_path, img_dir, image_ids=None, patch_size=512, stride=256, space_threshold=0.9, patches_per_image=10, is_val=False):
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.stride = stride
        self.space_threshold = space_threshold
        self.patches_per_image = patches_per_image
        self.is_val = is_val

        with open(json_path, 'r') as f:
            full_data = json.load(f)

        if image_ids is not None:
            self.images = [img for img in full_data['images'] if img['id'] in image_ids]
        else:
            self.images = full_data['images']

        self.full_annotations = full_data['annotations']
        self.ann_map = self._build_ann_map()

        # Scan for all valid patches
        self.valid_patches = self._find_all_valid_patches()

        # Only keep images that have at least one valid patch
        self.images = [img for img in self.images if img['id'] in self.valid_patches]

        # For validation, we pre-calculate a STATIC list of patches to ensure consistency
        if self.is_val:
            self.static_patches = self._generate_static_patch_list()
        else:
            self.static_patches = None

    def _build_ann_map(self):
        ann_map = {}
        for ann in self.full_annotations:
            img_id = ann['image_id']
            if img_id not in ann_map:
                ann_map[img_id] = []
            ann_map[img_id].append(ann)
        return ann_map

    def _find_all_valid_patches(self):
        valid_patches = {}
        for img in self.images:
            img_id = img['id']
            fname = img['file_name']
            img_path = os.path.join(self.img_dir, fname)
            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            from utils.preprocessing import get_disk_mask
            disk_mask = get_disk_mask(image)

            img_valid_coords = []
            for y in range(0, 2048 - self.patch_size + 1, self.stride):
                for x in range(0, 2048 - self.patch_size + 1, self.stride):
                    patch_mask = disk_mask[y:y+self.patch_size, x:x+self.patch_size]
                    space_ratio = 1.0 - (np.sum(patch_mask == 255) / (self.patch_size**2))

                    if space_ratio < self.space_threshold:
                        img_valid_coords.append((x, y))

            if img_valid_coords:
                valid_patches[img_id] = img_valid_coords
        return valid_patches

    def _generate_static_patch_list(self):
        static_list = []
        for img in self.images:
            img_id = img['id']
            coords = self.valid_patches[img_id]
            sampled = coords[:self.patches_per_image]
            for x, y in sampled:
                static_list.append((img_id, x, y))
        return static_list

    def __len__(self):
        if self.is_val:
            return len(self.static_patches)
        return len(self.images) * self.patches_per_image

    def __getitem__(self, idx):
        if self.is_val:
            img_id, x, y = self.static_patches[idx]
            img = next(i for i in self.images if i['id'] == img_id)
            fname = img['file_name']
        else:
            img_idx = idx // self.patches_per_image
            img = self.images[img_idx]
            img_id = img['id']
            fname = img['file_name'] # Wait, I'll fix this typo
            coords = self.valid_patches[img_id]
            x, y = random.choice(coords)

        img_path = os.path.join(self.img_dir, fname)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        enhanced, disk_mask = preprocess_image(image)

        img_patch = enhanced[y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = disk_mask[y:y+self.patch_size, x:x+self.patch_size]

        combined_mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint8)
        anns = self.ann_map.get(img_id, [])
        for ann in anns:
            full_mask = polygon_to_mask(ann['segmentation'], shape=(2048, 2048))
            inst_patch = full_mask[y:y+self.patch_size, x:x+self.patch_size]
            combined_mask = np.maximum(combined_mask, inst_patch)

        img_tensor = torch.from_numpy(np.stack([img_patch]*3, axis=0)).float() / 255.0
        disk_tensor = torch.from_numpy(mask_patch).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(combined_mask).float().unsqueeze(0)

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'disk_mask': disk_tensor,
            'image_id': img_id,
            'patch_coords': (x, y)
        }

def split_solar_dataset(data, train_ratio=0.8):
    obs_to_imgs = defaultdict(list)
    for img in data['images']:
        obs = img['file_name'].split('.')[0][-2:]
        obs_to_imgs[obs].append(img['id'])

    train_ids, val_ids = [], []
    for obs, ids in obs_to_imgs.items():
        np.random.shuffle(ids)
        split_idx = int(len(ids) * train_ratio)
        train_ids.extend(ids[:split_idx])
        val_ids.extend(ids[split_idx:])
    return train_ids, val_ids
