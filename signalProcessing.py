import numpy as np
import matplotlib.pyplot as plt

def Range_Doppler(data, mode=0, padding_size=None):
    window_1 = np.reshape(np.hanning(data.shape[1]), (data.shape[1], -1))
    window_2 = np.reshape(np.hanning(data.shape[0]), (-1, data.shape[0]))
    window = window_1 * window_2
    channel = data.shape[2]
    for i in range(channel):
        data[:, :, i] = data[:, :, i] * window.T

    if mode == 0:
        return np.fft.fft2(data, s=padding_size, axes=[0, 1])
    elif mode == 1:
        rdi_abs = np.fft.fft2(data, s=padding_size, axes=[0, 1])
        rdi_abs = np.transpose(np.fft.fftshift(np.abs(rdi_abs), axes=0), [1, 0, 2])
        return rdi_abs
    elif mode == 2:
        rdi_raw = np.fft.fft2(data, s=padding_size, axes=[0, 1])
        rdi_abs = np.transpose(np.fft.fftshift(np.abs(rdi_raw), axes=0), [1, 0, 2])
        return [rdi_raw, rdi_abs]
    else:
        raise ValueError("Error mode")

def calculate_range_resolution(adc_samples, sample_rate_ksps, freq_slope_mhz_us):
    c = 3e8
    fs = sample_rate_ksps * 1e3
    slope = freq_slope_mhz_us * 1e12
    sampling_time = adc_samples / fs
    bandwidth = slope * sampling_time
    res_meters = c / (2 * bandwidth)
    return res_meters, bandwidth

file_path = 'rawData/500_32_64_objectIn350.bin'
num_frames = 500
num_chirps = 32
num_rx = 4
num_samples = 64

sample_rate = 2000
frequency_slope = 65.998

resolution, bw = calculate_range_resolution(num_samples, sample_rate, frequency_slope)

print(f"Bandwidth Used: {bw/1e9:.2f} GHz")
print(f"Range Resolution: {resolution*100:.2f} cm")

ints_per_frame = num_chirps * num_rx * num_samples * 2

try:
    full_adc_data = np.fromfile(file_path, dtype=np.int16)
    framewise_raw = full_adc_data.reshape(num_frames, ints_per_frame)
    first_frame_raw = framewise_raw[0]
except FileNotFoundError:
    print(f"ERROR: File '{file_path}' not found. Please check the path.")
    exit()

frameData = np.reshape(first_frame_raw, [-1, 4])
frameData = frameData[:, 0:2:] + 1j * frameData[:, 2::]
frameData = np.reshape(frameData, [num_chirps, -1, num_samples])
frameData = frameData.transpose([0, 2, 1])

padding = [128, 64]
rdi = Range_Doppler(frameData, mode=1, padding_size=padding)
rdi_img = rdi[:, :, 0]

flattened = rdi_img.flatten()
top_3_flat_indices = np.argsort(flattened)[-3:][::-1]
top_rows, top_cols = np.unravel_index(top_3_flat_indices, rdi_img.shape)

print("\n--- TOP 3 DETECTED PEAKS ---")
peaks_info = []
for i in range(3):
    r_bin = top_rows[i]
    d_bin = top_cols[i]
    strength = rdi_img[r_bin, d_bin]
    
    distance_m = r_bin * resolution
    
    print(f"Signal #{i+1}: Strength={strength:.1f} | Distance: {distance_m:.3f} m (Bin: {r_bin}) | Doppler Bin: {d_bin}")
    peaks_info.append((d_bin, distance_m))

max_range = resolution * padding[1]
doppler_bins = padding[0]

plt.figure(figsize=(9, 7))

plt.imshow(rdi_img, aspect='auto', origin='lower', cmap='jet',
           extent=[0, doppler_bins, 0, max_range])


plt.title(f"Range-Doppler Heatmap & Top 3 Peaks\n(Max: {peaks_info[0][1]:.2f}m)")
plt.xlabel("Doppler Bins")
plt.ylabel("Distance (Meters)")
plt.colorbar(label="Signal Strength (Magnitude)")
plt.grid(alpha=0.3, color='white', linestyle='--')
plt.show()