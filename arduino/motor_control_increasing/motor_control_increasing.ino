/*
 * Motor Control for mmWave Recording
 * 
 * This Arduino sketch controls a vibration motor via L298N driver.
 * It receives commands from the Lua script over serial:
 *   'M' or '1' = Motor ON
 *   'S' or '0' = Motor STOP
 *   '?' = Query status
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

// Motor speed settings
const int START_SPEED = 64;
const int TARGET_SPEED = 255;
const unsigned long RAMP_DURATION = 2000; // 2000ms = 2 seconds

// LED for status indication
const int STATUS_LED = 13;

// Motor state
bool motorRunning = false;
bool isRamping = false;
unsigned long rampStartTime = 0;

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
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    switch(command) {
      case 'M':  // Motor ON
      case '1':
        startMotor();
        Serial.println("MOTOR_ON");
        break;
        
      case 'S':  // Motor STOP
      case '0':
        stopMotor();
        Serial.println("MOTOR_OFF");
        break;
        
      case '?':  // Query status
        if(motorRunning) {
          Serial.println("STATUS:ON");
        } else {
          Serial.println("STATUS:OFF");
        }
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

  // Handle Motor Speed Ramping
  if (motorRunning && isRamping) {
    unsigned long currentTime = millis();
    unsigned long elapsed = currentTime - rampStartTime;

    if (elapsed < RAMP_DURATION) {
      // Linearly increase speed from START_SPEED to TARGET_SPEED
      int currentSpeed = map(elapsed, 0, RAMP_DURATION, START_SPEED, TARGET_SPEED);
      analogWrite(MOTOR_ENA, currentSpeed);
    } else {
      // Ramp finished, maintain target speed
      analogWrite(MOTOR_ENA, TARGET_SPEED);
      isRamping = false;
    }
  }
}

void startMotor() {
  // Set direction (forward)
  digitalWrite(MOTOR_IN1, HIGH);
  digitalWrite(MOTOR_IN2, LOW);
  
  // Set initial speed
  analogWrite(MOTOR_ENA, START_SPEED);
  
  // Status LED ON
  digitalWrite(STATUS_LED, HIGH);
  
  motorRunning = true;
  
  // Initialize Ramping
  isRamping = true;
  rampStartTime = millis();
}

void stopMotor() {
  // Disable motor
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  analogWrite(MOTOR_ENA, 0);
  
  // Status LED OFF
  digitalWrite(STATUS_LED, LOW);
  
  motorRunning = false;
  isRamping = false;
}
