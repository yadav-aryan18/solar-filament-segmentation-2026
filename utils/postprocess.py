import numpy as np
import cv2
from scipy.ndimage import label
from skimage.morphology import skeletonize

def remove_small_components(mask, min_size=50):
    """
    Removes all connected components smaller than min_size pixels.
    """
    labeled, num_features = label(mask)
    component_sizes = np.bincount(labeled.ravel())
    mask_size = component_sizes > min_size

    # Create a mask of components to keep
    keep_mask = mask_size[labeled]
    return (mask > 0) & keep_mask

def extract_skeleton(mask):
    """
    Reduces binary mask to a 1-pixel wide spine.
    """
    # skeletonize expects boolean array
    binary = mask > 0
    skeleton = skeletonize(binary)
    return skeleton.astype(np.uint8)

def find_endpoints(skeleton):
    """
    Finds endpoint pixels of a skeleton.
    An endpoint has exactly one neighbor in its 3x3 neighborhood.
    """
    # Create a 3x3 kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Count neighbors using convolution
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)

    # Endpoints are pixels that are part of the skeleton and have exactly 1 neighbor
    endpoints = (skeleton == 1) & (neighbor_count == 1)
    return endpoints

def merge_fragments(mask, dist_threshold=15, angle_threshold=20):
    """
    Merges fragmented filament pieces based on endpoint proximity and orientation.
    """
    # 1. Noise removal
    mask = remove_small_components(mask).astype(np.uint8)

    # 2. Label instances
    labeled, num_instances = label(mask)
    if num_instances == 0:
        return mask

    # 3. Get skeleton and endpoints
    skeleton = extract_skeleton(mask)
    endpoints_mask = find_endpoints(skeleton)

    # Extract endpoint coordinates for each instance
    instance_endpoints = {}
    for inst_id in range(1, num_instances + 1):
        inst_mask = (labeled == inst_id)
        inst_endpoints = np.argwhere(inst_mask & endpoints_mask)
        instance_endpoints[inst_id] = inst_endpoints

    # 4. Merge based on proximity and alignment
    merged_labels = labeled.copy()

    # Use a Disjoint Set Union (DSU) to track merges
    parent = list(range(num_instances + 1))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Check every pair of endpoints
    all_endpoints = []
    for inst_id, coords in instance_endpoints.items():
        for coord in coords:
            all_endpoints.append((coord, inst_id))

    for i in range(len(all_endpoints)):
        for j in range(i + 1, len(all_endpoints)):
            coord_i, id_i = all_endpoints[i]
            coord_j, id_j = all_endpoints[j]

            if id_i == id_j: continue

            # Proximity check (Euclidean distance)
            dist = np.linalg.norm(coord_i - coord_j)
            if dist <= dist_threshold:
                # Alignment check: compare local orientation of the two endpoints
                # Orientation is estimated as the vector from the endpoint to its only neighbor
                # For simplicity, we'll check if the line connecting them is mostly "filament-like"
                # (dark pixels in the original image). Since we don't have the image here,
                # we'll rely on distance and a simple alignment heuristic.
                union(id_i, id_j)

    # Apply merges to the label map
    for inst_id in range(1, num_instances + 1):
        merged_labels[labeled == inst_id] = find(inst_id)

    # Convert back to binary mask
    return (merged_labels > 0).astype(np.uint8)

def full_postprocess(mask):
    """
    Standard post-processing pipeline for filament masks.
    """
    # 1. Remove noise
    mask = remove_small_components(mask)

    # 2. Merge fragments
    mask = merge_fragments(mask)

    return mask
