import os
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from spectrogram_generator import generate_spectrogram


def process_all_spectrograms(
    input_base_dir="pipe-vibration",
    output_base_dir="spectrograms-output",
    num_frames=25,
    num_rx=4,
    num_chirps=64,
    num_samples=256,
    chirp_period=20,
    n_fft=4096,
    hop_length=256
):
    """
    Process all radar recordings in the specified directories and save spectrograms.
    
    Parameters:
    -----------
    input_base_dir : str
        Base directory containing 'closed' and 'full-speed' subdirectories
    output_base_dir : str
        Directory to save the generated spectrograms
    """
    # Define subdirectories to process
    subdirs = ["closed", "full-speed"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Statistics
    total_processed = 0
    total_failed = 0
    
    # Process each subdirectory
    for subdir in subdirs:
        input_dir = os.path.join(input_base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"⚠️  Warning: Directory {input_dir} does not exist, skipping...")
            continue
        
        # Create output subdirectory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all .bin files
        pattern = os.path.join(input_dir, "*.bin")
        bin_files = glob.glob(pattern)
        
        if not bin_files:
            print(f"⚠️  Warning: No .bin files found in {input_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {subdir}")
        print(f"Found {len(bin_files)} files")
        print(f"{'='*60}\n")
        
        # Process each file
        for file_path in tqdm(bin_files, desc=f"Processing {subdir}", unit="file"):
            try:
                # Generate spectrogram
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
                
                # Create output filename
                base_name = Path(file_path).stem  # Get filename without extension
                output_file = os.path.join(output_dir, f"{base_name}_spectrogram.npy")
                
                # Save spectrogram as numpy array
                np.save(output_file, S_db)
                
                total_processed += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {file_path}: {str(e)}")
                total_failed += 1
                continue
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {total_processed} files")
    print(f"❌ Failed: {total_failed} files")
    print(f"📁 Output directory: {output_base_dir}")
    print(f"{'='*60}\n")
    
    # Save metadata
    metadata = {
        "total_processed": total_processed,
        "total_failed": total_failed,
        "num_frames": num_frames,
        "num_rx": num_rx,
        "num_chirps": num_chirps,
        "num_samples": num_samples,
        "chirp_period": chirp_period,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "input_base_dir": input_base_dir,
        "subdirectories": subdirs
    }
    
    metadata_file = os.path.join(output_base_dir, "processing_metadata.npy")
    np.save(metadata_file, metadata)
    print(f"📝 Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    # Run batch processing with specified configuration
    process_all_spectrograms(
        input_base_dir="pipe-vibration",
        output_base_dir="spectrograms-output",
        num_frames=25,
        num_rx=4,
        num_chirps=64,
        num_samples=256,
        chirp_period=20,
        n_fft=4096,
        hop_length=256
    )
