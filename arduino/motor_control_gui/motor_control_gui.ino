/*
 * Motor Control with Multiple Vibration Modes
 * 
 * This Arduino sketch controls a vibration motor via L298N driver.
 * It receives commands over serial to switch between vibration modes:
 *   'C' = Constant vibration (steady speed)
 *   'I' = Increasing vibration (ramp up)
 *   'D' = Decreasing vibration (ramp down)
 *   'N' or 'S' or '0' = No vibration (stop motor)
 *   '?' = Query current mode
 * 
 * L298N Connections:
 *   - IN1 -> Arduino Pin 9 (PWM capable)
 *   - IN2 -> Arduino Pin 8
 *   - ENA -> Arduino Pin 10 (PWM capable)
 *   - GND -> Arduino GND
 */

// Pin definitions for L298N
const int MOTOR_IN1 = 9;     // Direction pin 1
const int MOTOR_IN2 = 8;     // Direction pin 2
const int MOTOR_ENA = 10;    // Enable/PWM pin

// LED for status indication
const int STATUS_LED = 13;

// Motor speed settings
const int CONSTANT_SPEED = 200;   // Speed for constant vibration
const int MIN_SPEED = 64;         // Minimum speed for ramping
const int MAX_SPEED = 255;        // Maximum speed for ramping
const unsigned long RAMP_DURATION = 4000; // 4 seconds for full ramp

// Vibration modes
enum VibrationMode {
  MODE_NONE = 0,
  MODE_CONSTANT = 1,
  MODE_INCREASING = 2,
  MODE_DECREASING = 3
};

// Current state
VibrationMode currentMode = MODE_NONE;
bool motorRunning = false;
unsigned long modeStartTime = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Set motor control pins as outputs
  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(MOTOR_ENA, OUTPUT);
  pinMode(STATUS_LED, OUTPUT);
  
  // Ensure motor is OFF at startup
  stopMotor();
  
  // Flash LED to indicate ready
  for(int i = 0; i < 3; i++) {
    digitalWrite(STATUS_LED, HIGH);
    delay(100);
    digitalWrite(STATUS_LED, LOW);
    delay(100);
  }
  
  Serial.println("MOTOR_CONTROL_READY");
}

void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    switch(command) {
      case 'C':  // Constant vibration
      case 'c':
        setMode(MODE_CONSTANT);
        Serial.println("MODE:CONSTANT");
        break;
        
      case 'I':  // Increasing vibration
      case 'i':
        setMode(MODE_INCREASING);
        Serial.println("MODE:INCREASING");
        break;
        
      case 'D':  // Decreasing vibration
      case 'd':
        setMode(MODE_DECREASING);
        Serial.println("MODE:DECREASING");
        break;
        
      case 'N':  // No vibration
      case 'n':
      case 'S':
      case 's':
      case '0':
        setMode(MODE_NONE);
        Serial.println("MODE:NONE");
        break;
        
      case '?':  // Query current mode
        printCurrentMode();
        break;
        
      case '\n':  // Ignore newlines
      case '\r':
        break;
        
      default:
        Serial.print("UNKNOWN:");
        Serial.println(command);
        break;
    }
  }
  
  // Update motor based on current mode
  updateMotor();
}

void setMode(VibrationMode mode) {
  currentMode = mode;
  modeStartTime = millis();
  
  if (mode == MODE_NONE) {
    stopMotor();
  } else {
    // Start motor with appropriate initial speed
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, LOW);
    digitalWrite(STATUS_LED, HIGH);
    motorRunning = true;
    
    // Set initial speed based on mode
    switch(mode) {
      case MODE_CONSTANT:
        analogWrite(MOTOR_ENA, CONSTANT_SPEED);
        break;
      case MODE_INCREASING:
        analogWrite(MOTOR_ENA, MIN_SPEED);
        break;
      case MODE_DECREASING:
        analogWrite(MOTOR_ENA, MAX_SPEED);
        break;
      default:
        break;
    }
  }
}

void updateMotor() {
  if (!motorRunning) return;
  
  unsigned long elapsed = millis() - modeStartTime;
  int currentSpeed;
  
  switch(currentMode) {
    case MODE_CONSTANT:
      // Maintain constant speed
      analogWrite(MOTOR_ENA, CONSTANT_SPEED);
      break;
      
    case MODE_INCREASING:
      // Ramp up speed
      if (elapsed < RAMP_DURATION) {
        currentSpeed = map(elapsed, 0, RAMP_DURATION, MIN_SPEED, MAX_SPEED);
        analogWrite(MOTOR_ENA, currentSpeed);
      } else {
        analogWrite(MOTOR_ENA, MAX_SPEED);
      }
      break;
      
    case MODE_DECREASING:
      // Ramp down speed
      if (elapsed < RAMP_DURATION) {
        currentSpeed = map(elapsed, 0, RAMP_DURATION, MAX_SPEED, MIN_SPEED);
        analogWrite(MOTOR_ENA, currentSpeed);
      } else {
        analogWrite(MOTOR_ENA, MIN_SPEED);
      }
      break;
      
    default:
      break;
  }
}

void stopMotor() {
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  analogWrite(MOTOR_ENA, 0);
  digitalWrite(STATUS_LED, LOW);
  motorRunning = false;
  currentMode = MODE_NONE;
}

void printCurrentMode() {
  switch(currentMode) {
    case MODE_NONE:
      Serial.println("MODE:NONE");
      break;
    case MODE_CONSTANT:
      Serial.println("MODE:CONSTANT");
      break;
    case MODE_INCREASING:
      Serial.println("MODE:INCREASING");
      break;
    case MODE_DECREASING:
      Serial.println("MODE:DECREASING");
      break;
  }
}
