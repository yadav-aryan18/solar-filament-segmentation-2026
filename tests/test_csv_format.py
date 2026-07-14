import cv2
import numpy as np
from utils.coco_utils import mask_to_rle

# Create a realistic filament-like mask
test_mask = np.zeros((2048, 2048), dtype=np.uint8)
pts = [(1000 + int(200*np.sin(i*0.3)), 1000 + int(50*np.cos(i*0.5))) for i in range(50)]
for x, y in pts:
    cv2.circle(test_mask, (x, y), 15, 1, -1)

# Generate the RLE using our fixed function
rle = mask_to_rle(test_mask)
print("RLE type:", type(rle))
print("RLE starts with 'b'?:", rle.startswith("b'"))
print()

# Simulate the manual CSV writing (same logic as inference.py)
rows = [{"filament_id": "20150125172714Mh_1", "segmentation_rle": rle},
        {"filament_id": "20150125172714Mh_2", "segmentation_rle": "^Vj02jo16I5O2O1"}]
with open("test_submission.csv", "w") as f:
    f.write("filament_id,segmentation_rle\n")
    for row in rows:
        r = row["segmentation_rle"]
        if isinstance(r, bytes):
            r = r.decode("utf-8")
        r = r.strip().strip('"').strip("'")
        f.write(f"{row['filament_id']},{r}\n")

# Verify the output
with open("test_submission.csv", "r") as f:
    content = f.read()

print("=== CSV HEAD ===")
print(content[:250])
print("=== Last 100 chars ===")
print(repr(content[-100:]))
print()
print("Contains b' prefix?:", "b'" in content)
print("Contains double quotes?:", '"' in content)
print("Header correct?:", content.startswith("filament_id,segmentation_rle"))

import os
os.remove("test_submission.csv")
print("\nTest passed!" if all([
    "b'" not in content,
    '"' not in content,
    content.startswith("filament_id,segmentation_rle")
]) else "\nTest FAILED!")
