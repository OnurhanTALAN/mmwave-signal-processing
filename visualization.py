"""
Visualization utilities for radar data analysis.
"""
import matplotlib.pyplot as plt


def plot_range_doppler_heatmap(rdi_img, padding, resolution, peaks_info=None, figsize=(9, 7)):
    """
    Plot a Range-Doppler heatmap with optional peak markers.
    
    Parameters:
    -----------
    rdi_img : numpy.ndarray
        2D Range-Doppler image to plot
    padding : list or tuple
        Padding size as [doppler_padding, range_padding]
    resolution : float
        Range resolution in meters
    peaks_info : list of dict, optional
        List of peak information dictionaries from find_top_peaks()
        Each dict should contain 'doppler_bin' and 'distance_m' keys
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (9, 7))
    """
    max_range = resolution * padding[1]
    doppler_bins = padding[0]
    
    plt.figure(figsize=figsize)
    
    # Display the Range-Doppler image
    plt.imshow(rdi_img, aspect='auto', origin='lower', cmap='jet',
               extent=[0, doppler_bins, 0, max_range])
    
    # Set title based on peaks info
    if peaks_info and len(peaks_info) > 0:
        max_distance = peaks_info[0]['distance_m']
        plt.title(f"Range-Doppler Heatmap & Top {len(peaks_info)} Peaks\n(Max: {max_distance:.2f}m)")
    else:
        plt.title("Range-Doppler Heatmap")
    
    plt.xlabel("Doppler Bins")
    plt.ylabel("Distance (Meters)")
    plt.colorbar(label="Signal Strength (Magnitude)")
    plt.grid(alpha=0.3, color='white', linestyle='--')
    plt.show()

