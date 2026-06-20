#!/usr/bin/env python3
"""
Utility functions for time series visualization.
"""

import os
import base64
import numpy as np
import matplotlib.pyplot as plt

def generate_image_from_timeseries(case_idx, timeseries, cols, fig_dir, save_image=False):
    """
    Plot each channel in its own subplot, save as JPG, return base64 string.
    
    Args:
        case_idx: Unique identifier for the figure
        timeseries: Time series data to visualize (list or numpy array)
        cols: Column names for the time series
        fig_dir: Directory to save the figure
        save_image: Whether to keep the saved image file (default: False)
        
    Returns:
        Base64 encoded string of the image
    """
    # Create directory if it doesn't exist
    os.makedirs(fig_dir, exist_ok=True)
    
    # Ensure consistent naming scheme
    path = os.path.join(fig_dir, f"{case_idx}.jpg")
    
    # Convert numpy array to list if needed
    if isinstance(timeseries, np.ndarray):
        # Handle different dimensions
        if len(timeseries.shape) == 1:
            # Single series
            timeseries_list = [timeseries.tolist()]
        else:
            # Multiple series
            timeseries_list = [series.tolist() for series in timeseries]
    else:
        timeseries_list = timeseries
    
    # Ensure we have column names
    if not cols or len(cols) != len(timeseries_list):
        cols = [f"Series {i+1}" for i in range(len(timeseries_list))]
    
    # Handle the case where we have multiple subplots
    n = len(timeseries_list)
    if n > 1:
        figsize = (6, 2 * n)
        # Create subplots with the determined figure size
        fig, axes = plt.subplots(n, 1, figsize=figsize)
        
        for ax, series, title in zip(axes, timeseries_list, cols):
            ax.plot(series, linewidth=2)
            ax.set_title(title, fontsize=10, fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot(timeseries_list[0], linewidth=2)
        ax.set_title(cols[0], fontsize=10, fontweight='bold')
    
    # Save the figure with consistent settings
    plt.tight_layout()
    plt.savefig(path, format='jpg', dpi=100)
    plt.close(fig)

    # Read the image and convert to base64
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Delete the image file if not saving images
    if not save_image:
        try:
            os.remove(path)
        except OSError:
            # Ignore errors if file cannot be deleted
            pass
    
    return img_b64