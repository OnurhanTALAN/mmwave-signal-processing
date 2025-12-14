from data_loader import load_radar_data
from radar_utils import calculate_range_resolution
from signal_processing_utils import range_Doppler, find_top_peaks
from visualization import plot_range_doppler_heatmap

file_path = 'rawData/500_32_64_objectIn110.bin'
num_frames = 500
num_chirps = 32
num_rx = 4
num_samples = 64
sample_rate = 2000
frequency_slope = 65.998

resolution, bw = calculate_range_resolution(num_samples, sample_rate, frequency_slope)

print(f"Number of frames: {num_frames}")
print(f"Number of chirps: {num_chirps}")
print(f"Number of rx: {num_rx}")
print(f"Number of samples: {num_samples}")
print(f"Sample rate: {sample_rate} MHz")
print(f"Frequency slope: {frequency_slope} MHz/us")
print(f"Bandwidth used: {bw/1e9:.2f} GHz")
print(f"Range resolution: {resolution*100:.2f} cm")

frameData = load_radar_data(file_path, num_frames, num_chirps, num_rx, num_samples)

if frameData is None:
    exit()

padding = [128, 64]
rdi = range_Doppler(frameData, mode=1, padding_size=padding)
rdi_img = rdi[:, :, 0]

print("\n--- TOP 3 DETECTED PEAKS ---")
peaks_info = find_top_peaks(rdi_img, resolution, num_peaks=3)

plot_range_doppler_heatmap(rdi_img, padding, resolution, peaks_info)