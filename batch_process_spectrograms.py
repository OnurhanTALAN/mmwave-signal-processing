import os
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from spectrogram_generator import generate_spectrogram


def process_all_spectrograms(
    input_base_dir="pipe-vibration",
    output_base_dir="spectrograms-output\\underground-pipe-vibration-2s",
    subdirs=("decreasing-vibration", "increasing-vibration", "no-vibration", "full-vibration"),
    num_frames=110,
    num_rx=4,
    num_chirps=64,
    num_samples=256,
    chirp_period=20,
    n_fft=4096,
    hop_length=256,
    trim_sec=0.15,
    max_freq=512
):
    """
    Batch spectrogram generation with array-level
    time and frequency trimming.
    """

    os.makedirs(output_base_dir, exist_ok=True)

    total_processed = 0
    total_failed = 0

    for subdir in subdirs:
        input_dir = os.path.join(input_base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)

        if not os.path.exists(input_dir):
            print(f"Warning: {input_dir} does not exist, skipping.")
            continue

        os.makedirs(output_dir, exist_ok=True)

        bin_files = glob.glob(os.path.join(input_dir, "*.bin"))

        if not bin_files:
            print(f"Warning: No .bin files found in {input_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {subdir}")
        print(f"Found {len(bin_files)} files")
        print(f"{'='*60}\n")

        for file_path in tqdm(bin_files, desc=f"{subdir}", unit="file"):
            try:
                # --- Generate full spectrogram ---
                S_db, Fs = generate_spectrogram(
                    file_path=file_path,
                    num_frames=num_frames,
                    num_rx=num_rx,
                    num_chirps=num_chirps,
                    num_samples=num_samples,
                    chirp_period=chirp_period,
                    n_fft=n_fft,
                    hop_length=hop_length
                )

                # --- Time trimming (array-level) ---
                frame_duration = hop_length / Fs
                trim_frames = int(trim_sec / frame_duration)

                S_db = S_db[:, trim_frames:-trim_frames]

                # --- Frequency trimming (0–max_freq Hz) ---
                freq_resolution = Fs / n_fft
                max_bin = int(max_freq / freq_resolution)

                S_db = S_db[:max_bin, :]

                # --- Save ---
                base_name = Path(file_path).stem
                output_file = os.path.join(
                    output_dir,
                    f"{base_name}_spectrogram.npy"
                )

                np.save(output_file, S_db)
                total_processed += 1

            except Exception as e:
                print(f"\nError processing {file_path}: {e}")
                total_failed += 1

    # --- Summary ---
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {total_processed}")
    print(f"Failed: {total_failed}")
    print(f"Output directory: {output_base_dir}")
    print(f"{'='*60}\n")

    # --- Metadata ---
    metadata = {
        "num_frames": num_frames,
        "num_rx": num_rx,
        "num_chirps": num_chirps,
        "num_samples": num_samples,
        "chirp_period": chirp_period,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "trim_sec": trim_sec,
        "max_freq": max_freq,
        "input_base_dir": input_base_dir,
        "subdirectories": subdirs,
        "total_processed": total_processed,
        "total_failed": total_failed
    }

    np.save(os.path.join(output_base_dir, "processing_metadata.npy"), metadata)

if __name__ == "__main__":
    process_all_spectrograms(
        input_base_dir="pipe-vibration/underground-pipe-vibration-2s",
        output_base_dir="spectrograms-output/underground-pipe-vibration-2s",
        subdirs=(
            "decreasing-vibration",
            "increasing-vibration",
            # "no-vibration",
            # "full-vibration"
        ),
        num_frames=110,
        num_rx=4,
        num_chirps=64,
        num_samples=256,
        chirp_period=20,
        n_fft=4096,
        hop_length=256,
        trim_sec=0.15,
        max_freq=512
    )
