import torch
import numpy as np
import cv2
from dataset.solar_dataset import SolarDataset
from utils.preprocessing import preprocess_image
from utils.coco_utils import polygon_to_mask
import os
import json

def test_dataset_alignment():
    print("Running Dataset Alignment Tests...")

    # 1. Setup synthetic data for a single image
    test_img_dir = "test_data"
    os.makedirs(test_img_dir, exist_ok=True)

    # Create a 2048x2048 image with a unique mark at (1000, 1000)
    img = np.zeros((2048, 2048), dtype=np.uint8)
    img[1000, 1000] = 255 # The landmark
    # Create a disk to pass the space_threshold
    cv2.circle(img, (1024, 1024), 800, 128, -1)
    cv2.imwrite(os.path.join(test_img_dir, "test.jpeg"), img)

    # Create a synthetic COCO JSON
    test_json = "test.json"
    # The mask should cover the landmark at (1000, 1000)
    # Polygon: [x, y, x, y, ...]
    poly = [999, 999, 1001, 999, 1001, 1001, 999, 1001, 999, 999]
    data = {
        "images": [{"id": "test_id", "file_name": "test.jpeg", "width": 2048, "height": 2048}],
        "annotations": [{"id": "ann_1", "image_id": "test_id", "segmentation": [poly], "category_id": 1}]
    }
    with open(test_json, 'w') as f:
        json.dump(data, f)

    try:
        # Initialize dataset
        ds = SolarDataset(test_json, test_img_dir, image_ids=["test_id"], patches_per_image=1)

        # Test 1: Length check
        assert len(ds) == 1, f"Expected len 1, got {len(ds)}"
        print("✅ Test 1: Length correct.")

        # Test 2: Shape check
        sample = ds[0]
        assert sample['image'].shape == (3, 512, 512), f"Wrong image shape: {sample['image'].shape}"
        assert sample['mask'].shape == (1, 512, 512), f"Wrong mask shape: {sample['mask'].shape}"
        assert sample['disk_mask'].shape == (1, 512, 512), f"Wrong disk_mask shape: {sample['disk_mask'].shape}"
        print("✅ Test 2: Tensor shapes correct.")

        # Test 3: Alignment check
        # We sample until we get a patch that contains our landmark at (1000, 1000)
        found_landmark = False
        for _ in range(100):
            sample = ds[0]
            x, y = sample['patch_coords']
            # Landmark is at (1000, 1000). Check if it's in the patch.
            if x <= 1000 < x + 512 and y <= 1000 < y + 512:
                # Calculate relative coordinates in the patch
                rel_x, rel_y = 1000 - x, 1000 - y

                # Check if image pixel is high and mask pixel is high
                img_val = sample['image'][0, rel_y, rel_x]
                mask_val = sample['mask'][0, rel_y, rel_x]

                if img_val > 0.5 and mask_val > 0.5:
                    found_landmark = True
                    break

        assert found_landmark, "Landmark and mask did not align in the patch!"
        print("✅ Test 3: Image-Mask alignment verified.")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_img_dir)
        os.remove(test_json)

if __name__ == "__main__":
    test_dataset_alignment()
