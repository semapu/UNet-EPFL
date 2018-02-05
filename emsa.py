
from itertools import product, chain

import numpy as np

def sample_boxes(shape, num_boxes,
            min_radius, max_radius,
            mask, random_state):
    
    if mask is None:
        mask = np.ones(shape, dtype=np.int_)
    
    if max_radius is None:
        max_radius = int(np.mean(shape) // 2)
    
    # Sample centers
    all_coords = np.argwhere(mask)
    indices = random_state.choice(len(all_coords), size=num_boxes)
    coords = all_coords[indices]
    
    # Sample radius
    radius = random_state.randint(low=min_radius, high=max_radius, size=num_boxes)
    
    # Create boxes
    lower_bounds = coords - radius[:, None]
    upper_bounds = coords + radius[:, None] + 1
    
    # Crop boxes
    lower_bounds[lower_bounds < 0] = 0
    for dim in range(3):
        upper_bounds[upper_bounds[:,dim] >= shape[dim], dim] = shape[dim] - 1
    
    return lower_bounds, upper_bounds

def boxes_corners(lower_bounds, upper_bounds):
    
    ndim = lower_bounds.shape[1]
    
    aux = np.array(list(product(*[[0,1]]*ndim))).T
    
    corners = []
    for i in range(ndim):
        corner_coords = np.array([lower_bounds[:,i], upper_bounds[:,i]])
        corners.append(corner_coords[aux[i]])
    
    return corners

def boxes_volume(lower_bounds, upper_bounds):
    
    return (upper_bounds - lower_bounds).prod(1)

def ersa(true_density, pred_density, mask,
            num_subarrays, min_radius, max_radius=None,
            random_state=np.random, return_boxes=False):
    """Estimation over random subarrays"""
    
    # Integral images
    ii_true = (true_density * mask).cumsum(0).cumsum(1).cumsum(2)
    ii_pred = (pred_density * mask).cumsum(0).cumsum(1).cumsum(2)
    
    boxes = sample_boxes(true_density.shape, num_subarrays,
                        min_radius, max_radius, mask, random_state)
    corners = boxes_corners(*boxes)
    volumes = boxes_volume(*boxes)
    
    a, b, c, d, e, f, g, h = ii_true[corners]
    true_sums = h - g - f - d + b + c + e - a
    
    a, b, c, d, e, f, g, h = ii_pred[corners]
    pred_sums = h - g - f - d + b + c + e - a
    
    if return_boxes:
        return true_sums, pred_sums, volumes, boxes
    
    return true_sums, pred_sums, volumes

def diff_vesicles_per_voxel(true_sums, pred_sums, volumes):
    return np.mean(np.abs(true_sums - pred_sums) / volumes)
