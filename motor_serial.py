import serial
import time
import sys
import os

SIGNAL_FILE = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\motor_signal.txt"

def main():
    if len(sys.argv) < 2:
        print("Usage: python motor_serial.py <COM_PORT>")
        sys.exit(1)

    com_port = sys.argv[1]
    arduino = serial.Serial(com_port, 9600, timeout=1)
    time.sleep(2)

    current_state = None
    running = True

    while running:
        try:
            if os.path.exists(SIGNAL_FILE):
                with open(SIGNAL_FILE, 'r') as f:
                    signal = f.read().strip().upper()

                if signal == "EXIT":
                    arduino.write(b'S')
                    break

                elif signal == "INC" and current_state != "INC":
                    print("[Motor] Increasing vibration")
                    arduino.write(b'I')
                    current_state = "INC"

                elif signal == "DEC" and current_state != "DEC":
                    print("[Motor] Decreasing vibration")
                    arduino.write(b'D')
                    current_state = "DEC"

                elif signal == "OFF" and current_state != "OFF":
                    print("[Motor] Motor OFF")
                    arduino.write(b'S')
                    current_state = "OFF"

            time.sleep(0.05)

        except IOError:
            time.sleep(0.01)

    arduino.close()

if __name__ == "__main__":
    main()
