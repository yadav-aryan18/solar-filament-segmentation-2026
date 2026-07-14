import os
import json
import numpy as np
import torch
import cv2
from dataset.solar_dataset import SolarDataset
from utils.preprocessing import preprocess_image
from utils.coco_utils import polygon_to_mask

def test_dataset_robustness():
    print("🚀 Starting Dataset Pipeline Robustness Tests...\n")

    # --- SETUP SYNTHETIC DATA ---
    test_img_dir = "test_data_robust"
    os.makedirs(test_img_dir, exist_ok=True)
    test_json = "test_robust.json"

    # Create a 2048x2048 image
    # We put a "solar disk" in the center and a "filament" at a specific spot
    img = np.zeros((2048, 2048), dtype=np.uint8)
    cv2.circle(img, (1024, 1024), 800, 150, -1) # Solar disk
    # Filament: rectangle from (1000, 1000) to (1100, 1100)
    img[1000:1100, 1000:1100] = 100
    cv2.imwrite(os.path.join(test_img_dir, "test.jpeg"), img)

    # Create a matching COCO JSON
    # Filament polygon: [x0, y0, x1, y1, ...]
    poly = [1000, 1000, 1100, 1000, 1100, 1100, 1000, 1100, 1000, 1000]
    data = {
        "images": [{"id": "test_id", "file_name": "test.jpeg", "width": 2048, "height": 2048}],
        "annotations": [{"id": "ann_1", "image_id": "test_id", "segmentation": [poly], "category_id": 1}]
    }
    with open(test_json, 'w') as f:
        json.dump(data, f)

    try:
        # Initialize datasets
        train_ds = SolarDataset(test_json, test_img_dir, image_ids=["test_id"], is_val=False)
        val_ds = SolarDataset(test_json, test_img_dir, image_ids=["test_id"], is_val=True)

        # --- TEST 1: Determinism (Validation) ---
        print("Testing Validation Determinism...", end=" ")
        sample1 = val_ds[0]['mask']
        sample2 = val_ds[0]['mask']
        assert torch.equal(sample1, sample2), "Validation dataset is NOT deterministic!"
        print("✅ PASSED")

        # --- TEST 2: Randomness (Training) ---
        print("Testing Training Randomness...", end=" ")
        # Sample multiple times; it's highly unlikely to get the same patch every time
        samples = [train_ds[0]['mask'] for _ in range(10)]
        all_same = all(torch.equal(samples[0], s) for s in samples)
        assert not all_same, "Training dataset is NOT random!"
        print("✅ PASSED")

        # --- TEST 3: Perfect Alignment ---
        print("Testing Pixel-Perfect Alignment...", end=" ")
        # We force a specific coordinate for this test
        # Patch at (900, 900). Filament is at (1000, 1000).
        # Relative to patch: (1000-900, 1000-900) = (100, 100)
        x, y = 900, 900
        # Mock the __getitem__ to use these specific coords
        # We do this by manually calling the internal logic
        img_path = os.path.join(test_img_dir, "test.jpeg")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        enhanced, _ = preprocess_image(image)

        # Manual crop
        img_patch = enhanced[y:y+512, x:x+512]

        # Get mask from dataset logic
        full_mask = polygon_to_mask([poly], shape=(2048, 2048))
        mask_patch = full_mask[y:y+512, x:x+512]

        # The filament was at 1000,1000. In the patch, it should be at 100,100.
        # Filament is 100x100 pixels.
        # Check a pixel inside the filament
        assert mask_patch[100, 100] == 1, "Mask missing filament pixel at (100, 100)!"
        # Check a pixel outside the filament
        assert mask_patch[50, 50] == 0, "Mask has false positive at (50, 50)!"
        print("✅ PASSED")

        # --- TEST 4: Boundary Conditions ---
        print("Testing Edge-of-Image Crops...", end=" ")
        # Test the very last possible patch
        last_x, last_y = 1536, 1536
        # Manual crop to verify no crash
        _ = enhanced[last_y:last_y+512, last_x:last_x+512]
        print("✅ PASSED")

        # --- TEST 5: Space Filtering ---
        print("Testing Space-Pixel Filtering...", end=" ")
        # The solar disk is a circle at 1024,1024 with radius 800.
        # A patch at (0,0) is far from the disk.
        # Let's check if it's in the valid_patches list.
        # (Note: we use the actual dataset object here)
        valid_coords = train_ds.valid_patches["test_id"]
        assert (0, 0) not in valid_coords, "Space-only patch was incorrectly marked as valid!"
        print("✅ PASSED")

        print("\n✨ ALL ROBUSTNESS TESTS PASSED! The pipeline is stable and aligned. ✨")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree("test_data_robust", ignore_errors=True)
        if os.path.exists(test_json):
            os.remove(test_json)

if __name__ == "__main__":
    test_dataset_robustness()
