"""
Arduino Motor Controller Module

Handles serial communication with Arduino for motor control.
Supports 4 vibration modes: constant, increasing, decreasing, none.
"""

import serial
import serial.tools.list_ports
import time
from typing import Optional


class MotorController:
    """Controls the vibration motor via Arduino serial communication."""
    
    # Command mapping
    COMMANDS = {
        'constant': 'C',
        'increasing': 'I',
        'decreasing': 'D',
        'none': 'N'
    }
    
    def __init__(self, port: Optional[str] = None, baudrate: int = 9600):
        """
        Initialize the motor controller.
        
        Parameters:
        -----------
        port : str, optional
            COM port for Arduino. If None, will auto-detect.
        baudrate : int
            Serial baudrate (default 9600)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial: Optional[serial.Serial] = None
        self._connected = False
        self._current_mode = 'none'
    
    def find_arduino_port(self) -> Optional[str]:
        """
        Auto-detect Arduino COM port.
        
        Returns:
        --------
        str or None : COM port name if found
        """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # Common Arduino identifiers
            if 'Arduino' in port.description or 'CH340' in port.description or 'USB' in port.description:
                return port.device
            # Also check VID/PID for common Arduino boards
            if port.vid == 0x2341 or port.vid == 0x1A86:  # Arduino or CH340
                return port.device
        return None
    
    def connect(self) -> bool:
        """
        Connect to the Arduino.
        
        Returns:
        --------
        bool : True if connected successfully
        """
        try:
            # Find port if not specified
            if self.port is None:
                self.port = self.find_arduino_port()
                if self.port is None:
                    print("[MotorController] No Arduino found!")
                    return False
            
            print(f"[MotorController] Connecting to {self.port}...")
            
            # Open serial connection
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=2
            )
            
            # Wait for Arduino to reset (it resets on serial connect)
            time.sleep(2)
            
            # Clear any startup messages
            while self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                print(f"[MotorController] Arduino: {line}")
            
            self._connected = True
            print(f"[MotorController] Connected to Arduino on {self.port}")
            return True
            
        except serial.SerialException as e:
            print(f"[MotorController] Connection error: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the Arduino."""
        if self.serial and self.serial.is_open:
            # Stop motor before disconnecting
            self.set_mode('none')
            time.sleep(0.1)
            self.serial.close()
        self._connected = False
        print("[MotorController] Disconnected")
    
    def is_connected(self) -> bool:
        """Check if connected to Arduino."""
        return self._connected and self.serial and self.serial.is_open
    
    def set_mode(self, mode: str) -> bool:
        """
        Set the vibration mode.
        
        Parameters:
        -----------
        mode : str
            One of: 'constant', 'increasing', 'decreasing', 'none'
            
        Returns:
        --------
        bool : True if command sent successfully
        """
        if not self.is_connected():
            print(f"[MotorController] Not connected! Cannot set mode: {mode}")
            return False
        
        mode = mode.lower()
        if mode not in self.COMMANDS:
            print(f"[MotorController] Invalid mode: {mode}")
            return False
        
        try:
            command = self.COMMANDS[mode]
            self.serial.write(f"{command}\n".encode())
            self._current_mode = mode
            
            # Read response
            time.sleep(0.1)
            if self.serial.in_waiting:
                response = self.serial.readline().decode('utf-8', errors='ignore').strip()
                print(f"[MotorController] Response: {response}")
            
            print(f"[MotorController] Mode set to: {mode}")
            return True
            
        except serial.SerialException as e:
            print(f"[MotorController] Error sending command: {e}")
            return False
    
    def get_current_mode(self) -> str:
        """Get the current vibration mode."""
        return self._current_mode
    
    def query_mode(self) -> Optional[str]:
        """
        Query the Arduino for current mode.
        
        Returns:
        --------
        str or None : Current mode from Arduino
        """
        if not self.is_connected():
            return None
        
        try:
            self.serial.write(b"?\n")
            time.sleep(0.1)
            
            if self.serial.in_waiting:
                response = self.serial.readline().decode('utf-8', errors='ignore').strip()
                if response.startswith("MODE:"):
                    return response.split(":")[1].lower()
            return None
            
        except serial.SerialException:
            return None


# Global instance
_controller_instance: Optional[MotorController] = None


def get_motor_controller() -> MotorController:
    """Get the global motor controller instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = MotorController()
    return _controller_instance


if __name__ == "__main__":
    # Test the motor controller
    controller = MotorController()
    
    if controller.connect():
        print("\nTesting motor modes...")
        
        print("\n1. Testing CONSTANT mode...")
        controller.set_mode('constant')
        time.sleep(3)
        
        print("\n2. Testing INCREASING mode...")
        controller.set_mode('increasing')
        time.sleep(5)
        
        print("\n3. Testing DECREASING mode...")
        controller.set_mode('decreasing')
        time.sleep(5)
        
        print("\n4. Stopping motor...")
        controller.set_mode('none')
        time.sleep(1)
        
        controller.disconnect()
        print("\nTest complete!")
    else:
        print("Could not connect to Arduino!")
