"""
Utility functions for radar signal processing calculations.
"""


def calculate_range_resolution(adc_samples, sample_rate_ksps, freq_slope_mhz_us):
    """
    Calculate the range resolution and bandwidth of a radar system.
    
    Parameters:
    -----------
    adc_samples : int
        Number of ADC samples per chirp
    sample_rate_ksps : float
        Sampling rate in kilo-samples per second (ksps)
    freq_slope_mhz_us : float
        Frequency slope in MHz per microsecond
    
    Returns:
    --------
    tuple : (res_meters, bandwidth)
        res_meters : float
            Range resolution in meters
        bandwidth : float
            Bandwidth in Hz
    """
    c = 3e8  # Speed of light in m/s
    fs = sample_rate_ksps * 1e3  # Convert ksps to samples per second
    slope = freq_slope_mhz_us * 1e12  # Convert MHz/us to Hz/s
    sampling_time = adc_samples / fs
    bandwidth = slope * sampling_time
    res_meters = c / (2 * bandwidth)
    return res_meters, bandwidth

