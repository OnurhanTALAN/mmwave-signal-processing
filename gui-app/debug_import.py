
import sys
import os

print(f"Current Working Directory: {os.getcwd()}")
print(f"__file__: {__file__}")
file_dir = os.path.dirname(os.path.abspath(__file__))
print(f"File Directory: {file_dir}")

parent_dir = os.path.join(file_dir, "..")
abs_parent_dir = os.path.abspath(parent_dir)
print(f"Parent Directory (to be added): {abs_parent_dir}")

sys.path.insert(0, abs_parent_dir)
print(f"sys.path[0]: {sys.path[0]}")

print("Attempting to import mmwave...")
try:
    import mmwave
    print(f"Success! mmwave imported from: {mmwave.__file__}")
    from mmwave.dataloader import DCA1000
    print("Success! DCA1000 imported")
except ImportError as e:
    print(f"Error importing mmwave: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("Attempting to import spectrogram_processor...")
try:
    import spectrogram_processor
    print("spectrogram_processor imported successfully")
except ImportError as e:
    print(f"Error importing spectrogram_processor: {e}")
