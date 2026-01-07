"""
Motor Control Helper Script
This script monitors a signal file and controls the Arduino motor via serial.

Usage:
    python motor_serial.py COM3       # Replace COM3 with your Arduino port

The Lua script writes to a signal file:
    - "ON" = Start motor
    - "OFF" = Stop motor
    - "EXIT" = Exit this script
"""

import serial
import time
import sys
import os

# Signal file path (same as in Lua script)
SIGNAL_FILE = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\motor_signal.txt"

def main():
    if len(sys.argv) < 2:
        print("Usage: python motor_serial.py <COM_PORT>")
        print("Example: python motor_serial.py COM3")
        sys.exit(1)
    
    com_port = sys.argv[1]
    
    print(f"[Motor Control] Connecting to Arduino on {com_port}...")
    
    try:
        # Open serial connection
        arduino = serial.Serial(com_port, 9600, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        
        # Read initial response
        while arduino.in_waiting:
            line = arduino.readline().decode('utf-8').strip()
            print(f"[Arduino] {line}")
        
        print(f"[Motor Control] Connected! Monitoring signal file...")
        print(f"[Motor Control] Signal file: {SIGNAL_FILE}")
        
        # Create initial signal file
        with open(SIGNAL_FILE, 'w') as f:
            f.write("READY")
        
        current_state = None
        running = True
        
        while running:
            try:
                # Read signal file
                if os.path.exists(SIGNAL_FILE):
                    with open(SIGNAL_FILE, 'r') as f:
                        signal = f.read().strip().upper()
                    
                    if signal == "EXIT":
                        print("[Motor Control] Exit signal received. Stopping motor...")
                        arduino.write(b'S')
                        running = False
                        break
                    
                    if signal == "ON" and current_state != "ON":
                        print("[Motor Control] Starting motor...")
                        arduino.write(b'M')
                        current_state = "ON"
                        
                        # Read response
                        time.sleep(0.1)
                        while arduino.in_waiting:
                            line = arduino.readline().decode('utf-8').strip()
                            print(f"[Arduino] {line}")
                    
                    elif signal == "OFF" and current_state != "OFF":
                        print("[Motor Control] Stopping motor...")
                        arduino.write(b'S')
                        current_state = "OFF"
                        
                        # Read response
                        time.sleep(0.1)
                        while arduino.in_waiting:
                            line = arduino.readline().decode('utf-8').strip()
                            print(f"[Arduino] {line}")
                
                # Small delay to prevent CPU overuse
                time.sleep(0.05)  # 50ms polling interval
                
            except IOError:
                # File might be locked by Lua, try again
                time.sleep(0.01)
                continue
        
    except serial.SerialException as e:
        print(f"[ERROR] Serial connection failed: {e}")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n[Motor Control] Interrupted by user. Stopping motor...")
        try:
            arduino.write(b'S')
        except:
            pass
    
    finally:
        try:
            arduino.close()
            print("[Motor Control] Serial connection closed.")
        except:
            pass
        
        # Cleanup signal file
        try:
            os.remove(SIGNAL_FILE)
        except:
            pass

if __name__ == "__main__":
    main()
