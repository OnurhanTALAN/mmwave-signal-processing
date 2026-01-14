"""
Radar Worker Module for Background Processing

Handles file-based communication with mmWaveStudio Lua script
and processes radar recordings.
"""

import os
import time
from PyQt6.QtCore import QObject, pyqtSignal, QThread, pyqtSlot

from config import SIGNAL_FILE, OUTPUT_DIR, MS_PER_RECORD
from spectrogram_processor import generate_spectrogram, prepare_for_model, get_spectrogram_for_display
from model_inference import get_model


class RadarWorker(QObject):
    """
    Worker class for background radar operations.
    
    Signals:
    --------
    started : Recording started
    progress : Progress update (message, percentage)
    spectrogram_ready : Spectrogram data ready for display
    prediction_ready : Model prediction results ready
    finished : Recording and processing complete
    error : Error occurred
    """
    
    started = pyqtSignal()
    progress = pyqtSignal(str, int)
    spectrogram_ready = pyqtSignal(dict)
    prediction_ready = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = False
        self._recording_index = 1
    
    @pyqtSlot()
    def start_recording(self):
        """Start a radar recording and process the result."""
        self._is_running = True
        self.started.emit()
        
        try:
            # Step 0: Check if Lua script is ready
            status_file = SIGNAL_FILE.replace("gui_signal.txt", "gui_status.txt")
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = f.read().strip()
                print(f"[RadarWorker] Lua script status: {status}")
            else:
                print("[RadarWorker] Warning: Status file not found - is Lua script running?")
            
            # Step 1: Send record signal to Lua script
            self.progress.emit("Kayıt sinyali gönderiliyor...", 10)
            output_file = self._send_record_signal()
            
            # Step 2: Wait for recording to complete
            self.progress.emit("Radar kaydı alınıyor...", 20)
            wait_time = (MS_PER_RECORD / 1000) + 3  # Add buffer time
            print(f"[RadarWorker] Waiting {wait_time} seconds for recording...")
            time.sleep(wait_time)
            
            # Step 3: Wait for file to be written
            self.progress.emit("Dosya bekleniyor...", 40)
            if not self._wait_for_file(output_file, timeout=30):
                error_msg = (
                    f"Dosya oluşturulmadı: {output_file}\n\n"
                    "Olası nedenler:\n"
                    "1. mmWaveStudio'da single-record.lua scripti çalışmıyor olabilir\n"
                    "2. Radar cihazı bağlı olmayabilir\n"
                    "3. DCA1000 kartı hazır olmayabilir"
                )
                self.error.emit(error_msg)
                return
            
            # Step 4: Generate spectrogram
            self.progress.emit("Spectrogram oluşturuluyor...", 60)
            print(f"[RadarWorker] Generating spectrogram from: {output_file}")
            S_db, Fs = generate_spectrogram(output_file)
            print(f"[RadarWorker] Spectrogram shape: {S_db.shape}, Fs: {Fs}")
            
            # Get display data
            display_data = get_spectrogram_for_display(S_db, Fs)
            self.spectrogram_ready.emit(display_data)
            print("[RadarWorker] Spectrogram emitted to GUI")
            
            # Step 5: Prepare for model and run inference
            self.progress.emit("Model inference yapılıyor...", 80)
            print("[RadarWorker] Getting model instance...")
            model = get_model()
            
            if not model.is_loaded():
                print("[RadarWorker] Model not loaded, loading now...")
                if not model.load():
                    self.error.emit("Model yüklenemedi!")
                    return
            
            print("[RadarWorker] Getting normalization stats...")
            mean, std = model.get_normalization_stats()
            print(f"[RadarWorker] Normalization - mean: {mean}, std: {std}")
            
            print("[RadarWorker] Preparing input for model...")
            model_input = prepare_for_model(S_db, mean, std)
            print(f"[RadarWorker] Model input shape: {model_input.shape}")
            
            print("[RadarWorker] Running model prediction...")
            prediction = model.predict(model_input)
            print(f"[RadarWorker] Prediction result: {prediction}")
            
            prediction['file_path'] = output_file
            self.prediction_ready.emit(prediction)
            print("[RadarWorker] Prediction emitted to GUI")
            
            # Done
            self.progress.emit("Tamamlandı!", 100)
            self._recording_index += 1
            
        except Exception as e:
            self.error.emit(f"Hata: {str(e)}")
        finally:
            self._is_running = False
            self.finished.emit()
    
    def _send_record_signal(self) -> str:
        """
        Write record signal to file for Lua script to read.
        
        Returns:
        --------
        str : Expected output file path (with _Raw_0.bin suffix as created by DCA1000)
        """
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(SIGNAL_FILE), exist_ok=True)
        
        # Generate base filename for Lua script (with .bin extension)
        base_file = os.path.join(OUTPUT_DIR, f"adc_data_{self._recording_index}.bin")
        
        # The actual file created by DCA1000 has '_Raw_0' suffix
        # e.g., adc_data_1.bin becomes adc_data_1_Raw_0.bin
        expected_output_file = os.path.join(OUTPUT_DIR, f"adc_data_{self._recording_index}_Raw_0.bin")
        
        # Write signal file with base path (Lua passes this to DCA1000)
        with open(SIGNAL_FILE, 'w') as f:
            f.write(f"RECORD\n{base_file}")
        
        print(f"[RadarWorker] Signal file written: {SIGNAL_FILE}")
        print(f"[RadarWorker] Lua will record to: {base_file}")
        print(f"[RadarWorker] Expected output (DCA1000 adds _Raw_0): {expected_output_file}")
        
        return expected_output_file
    
    def _wait_for_file(self, file_path: str, timeout: float = 30) -> bool:
        """
        Wait for a file to be created and have non-zero size.
        
        Parameters:
        -----------
        file_path : str
            Path to the file to wait for
        timeout : float
            Maximum time to wait in seconds
            
        Returns:
        --------
        bool : True if file exists and has content, False if timeout
        """
        start_time = time.time()
        last_size = 0
        stable_count = 0
        
        while time.time() - start_time < timeout:
            if os.path.exists(file_path):
                current_size = os.path.getsize(file_path)
                if current_size > 0:
                    # Check if file size is stable (writing complete)
                    if current_size == last_size:
                        stable_count += 1
                        if stable_count >= 3:  # Size stable for 1.5 seconds
                            return True
                    else:
                        stable_count = 0
                    last_size = current_size
            time.sleep(0.5)
        
        return False
    
    def stop(self):
        """Stop the current operation."""
        self._is_running = False
    
    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self._is_running


class RadarThread(QThread):
    """
    Thread wrapper for RadarWorker.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = RadarWorker()
        self.worker.moveToThread(self)
    
    def run(self):
        """Run the recording in thread."""
        self.worker.start_recording()
