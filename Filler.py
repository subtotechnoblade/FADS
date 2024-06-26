import numpy as np
from scipy import ndimage


def Color_Fill(input_array, start_coords, threshold=0):
    # Copy the array to avoid changing the original image
    output_array = np.copy(input_array)

    # Get the value at the start coordinates
    target_value = output_array[start_coords]

    # Create a mask where the target_value is located
    loss = np.sum(np.abs(output_array - target_value), axis=-1) / (255.0 * 3)
    mask = loss <= threshold

    # Use binary_fill_holes if necessary or another algorithm like ndimage.label
    labeled_array, num_features = ndimage.label(mask)
    feature_index = labeled_array[start_coords]
    return feature_index == labeled_array


fill_kernel = np.ones((3, 3), dtype=np.bool_)


def Line_Fill(input_array, start_coords, threshold=0):
    # Copy the array to avoid changing the original image
    output_array = np.copy(input_array)

    # Get the value at the start coordinates
    target_value = output_array[start_coords]

    # Create a mask where the target_value is located
    loss = np.sum(np.abs(output_array - target_value), axis=-1) / (255 * 3)
    mask = loss <= threshold

    # Use binary_fill_holes if necessary or another algorithm like ndimage.label
    labeled_array, num_features = ndimage.label(mask, fill_kernel)
    feature_index = labeled_array[start_coords]

    return labeled_array == feature_index

