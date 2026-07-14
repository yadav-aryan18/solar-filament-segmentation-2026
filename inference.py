import os
import numpy as np
import torch
import cv2
import csv
from torch.utils.data import DataLoader
from models.solar_segformer import SolarSegFormer
from utils.preprocessing import preprocess_image
from utils.coco_utils import mask_to_rle
from utils.postprocess import full_postprocess
from scipy.ndimage import label

class TiledInference:
    """
    Performs inference on full 2048x2048 images using a sliding window
    with overlap blending to avoid edge artifacts.
    """
    def __init__(self, model, device, patch_size=512, stride=384, overlap=128):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.overlap = overlap

    def _blending_window(self):
        """
        Creates a 2D Hann-like blending window to fade out edges of patches.
        Prevents hard seams when stitching.
        """
        hann_1d = np.hanning(self.patch_size)
        hann_2d = np.sqrt(np.outer(hann_1d, hann_1d))
        return hann_2d

    def infer_full_image(self, image):
        """
        Runs inference on a full 2048x2048 image and returns the stitched mask.

        Args:
            image: (2048, 2048) uint8 grayscale image.

        Returns:
            mask: (2048, 2048) uint8 binary mask.
        """
        # 1. Preprocess
        enhanced, disk_mask = preprocess_image(image)

        # 2. Accumulator for blending
        accumulator = np.zeros((2048, 2048), dtype=np.float32)
        weight_sum = np.zeros((2048, 2048), dtype=np.float32)

        # 3. Tiling
        patch_coords = []
        for y in range(0, 2048 - self.patch_size + 1, self.stride):
            for x in range(0, 2048 - self.patch_size + 1, self.stride):
                patch_coords.append((x, y))
        # Also cover the last possible tiles at the edges
        for y in [2048 - self.patch_size]:
            for x in range(0, 2048 - self.patch_size + 1, self.stride):
                patch_coords.append((x, y))
        for x in [2048 - self.patch_size]:
            for y in range(0, 2048 - self.patch_size + 1, self.stride):
                patch_coords.append((x, y))

        # 4. Run inference on all patches
        self.model.eval()
        blend_window = self._blending_window()

        with torch.no_grad():
            for x, y in patch_coords:
                # Extract and preprocess patch
                img_patch = enhanced[y:y+self.patch_size, x:x+self.patch_size]

                # Convert to 3-channel tensor
                img_tensor = torch.from_numpy(
                    np.stack([img_patch]*3, axis=0)
                ).float().unsqueeze(0).to(self.device) / 255.0

                # Predict
                logits = self.model(img_tensor)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

                # Blend into accumulator
                accumulator[y:y+self.patch_size, x:x+self.patch_size] += pred * blend_window
                weight_sum[y:y+self.patch_size, x:x+self.patch_size] += blend_window

        # 5. Normalize and threshold
        final_mask = (accumulator / (weight_sum + 1e-6)) > 0.5

        # 6. Apply disk mask to remove space pixels
        final_mask = final_mask & (disk_mask == 255)

        return final_mask.astype(np.uint8)

def extract_instances(mask):
    """
    Extracts individual filament instances from a semantic mask using
    connected component analysis.

    Returns:
        list of np.ndarray: A list of binary masks, one per filament.
    """
    labeled, num_features = label(mask)
    instances = []
    for i in range(1, num_features + 1):
        instances.append((labeled == i).astype(np.uint8))
    return instances

def generate_submission_csv(image_dir, model_path, output_csv="submission.csv"):
    """
    Full inference pipeline: Loads model, processes all test images,
    extracts instances, and generates the submission CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SolarSegFormer(model_name="nvidia/mit-b0", num_labels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    inferencer = TiledInference(model, device)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpeg')])

    rows = []
    for idx, fname in enumerate(image_files):
        print(f"Processing {idx+1}/{len(image_files)}: {fname}")
        img_path = os.path.join(image_dir, fname)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Full inference
        semantic_mask = inferencer.infer_full_image(image)

        # Post-processing
        cleaned_mask = full_postprocess(semantic_mask)

        # Extract instances
        instances = extract_instances(cleaned_mask)

        # Encode each instance as RLE
        base_id = fname.split('.')[0]
        for inst_num, inst_mask in enumerate(instances):
            rle = mask_to_rle(inst_mask)
            filament_id = f"{base_id}_{inst_num+1}"
            rows.append({"filament_id": filament_id, "segmentation_rle": rle})

    # Write CSV manually to ensure exact competition format:
    # No quotes, no b'' prefix, just filament_id,rle_counts
    with open(output_csv, 'w') as f:
        f.write("filament_id,segmentation_rle\n")
        for row in rows:
            # Ensure rle is a clean string (no b'...' wrapping, no quotes)
            rle = row["segmentation_rle"]
            if isinstance(rle, bytes):
                rle = rle.decode('utf-8')
            # Strip any accidental quotes
            rle = rle.strip().strip('"').strip("'")
            f.write(f"{row['filament_id']},{rle}\n")

    print(f"Submission CSV saved to {output_csv} ({len(rows)} filaments)")

if __name__ == "__main__":
    DATA_DIR = '/media/nightking/WD-SN570/deep_learning_projects/solar_filament_segmentation_challenge_2026/dataset/MAGFiLO_1.0_Kaggle_2026'
    TEST_IMG_DIR = os.path.join(DATA_DIR, 'test/test_images')
    MODEL_PATH = "best_model.pth"

    generate_submission_csv(TEST_IMG_DIR, MODEL_PATH)