import torch
import numpy as np
import cv2
from utils.preprocessing import preprocess_image
from utils.coco_utils import polygon_to_mask

class SolarMapper:
    """
    DatasetMapper for Solar Filament Segmentation.
    Converts a dataset dict into the format required by Mask2Former.
    """
    def __init__(self, img_dir, patch_size=512):
        self.img_dir = img_dir
        self.patch_size = patch_size

    def __call__(self, dataset_dict):
        # 1. Load Image
        fname = dataset_dict["file_name"]
        img_path = os.path.join(self.img_dir, fname)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # 2. Preprocess (Disk Mask + CLAHE)
        enhanced, disk_mask = preprocess_image(image)

        # 3. Handle Patching (Simplification: we assume the DatasetCatalog
        # already handles the patch indices, or we apply it here)
        # For Mask2Former, we usually pass the full image or a pre-cropped patch.
        # To match the la-plan, we'll assume the input image is already the patch.
        # If it's the full image, we'd crop here.

        # For now, let's assume we are working with the enhanced image
        # Convert to RGB (3 channels) for Mask2Former
        image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        # 4. Convert Annotations to Masks
        # Mask2Former expects 'instances' as a list of dicts
        instances = []
        for ann in dataset_dict.get("annotations", []):
            mask = polygon_to_mask(ann['segmentation'], shape=image.shape)
            instances.append({
                "mask": torch.from_numpy(mask).bool(),
                "category_id": ann['category_id']
            })

        # 5. Construct the final dict
        dataset_dict["image"] = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        dataset_dict["disk_mask"] = torch.from_numpy(disk_mask).bool()
        dataset_dict["instances"] = instances

        return dataset_dict
