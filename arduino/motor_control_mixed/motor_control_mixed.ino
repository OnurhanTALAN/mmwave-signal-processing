/*
 * Unified Motor Control for mmWave Recording
 * Supports INCREASING and DECREASING vibration profiles
 *
 * Commands over Serial:
 *   'I' = Increasing vibration
 *   'D' = Decreasing vibration
 *   'S' = Motor STOP
 *   '?' = Query status
 */

const int MOTOR_IN1 = 9;
const int MOTOR_IN2 = 8;
const int MOTOR_ENA = 10;
const int STATUS_LED = 13;

// Speed profiles
const int INC_START_SPEED = 255;
const int INC_TARGET_SPEED = 64;

const int DEC_START_SPEED = 64;
const int DEC_TARGET_SPEED = 255;

const unsigned long RAMP_DURATION = 2200;

// State
bool motorRunning = false;
bool isRamping = false;
bool increasingMode = true;

unsigned long rampStartTime = 0;
int startSpeed = 0;
int targetSpeed = 0;

void setup() {
  Serial.begin(9600);

  pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);
  pinMode(MOTOR_ENA, OUTPUT);
  pinMode(STATUS_LED, OUTPUT);

  stopMotor();

  for (int i = 0; i < 3; i++) {
    digitalWrite(STATUS_LED, HIGH);
    delay(100);
    digitalWrite(STATUS_LED, LOW);
    delay(100);
  }

  Serial.println("MOTOR_CONTROL_READY");
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    switch (cmd) {
      case 'I':   // Increasing
        configureIncreasing();
        startMotor();
        Serial.println("MODE:INCREASING");
        break;

      case 'D':   // Decreasing
        configureDecreasing();
        startMotor();
        Serial.println("MODE:DECREASING");
        break;

      case 'S':
        stopMotor();
        Serial.println("MOTOR_OFF");
        break;

      case '?':
        Serial.println(motorRunning ? "STATUS:ON" : "STATUS:OFF");
        break;
    }
  }

  handleRamping();
}

void configureIncreasing() {
  increasingMode = true;
  startSpeed = INC_START_SPEED;
  targetSpeed = INC_TARGET_SPEED;
}

void configureDecreasing() {
  increasingMode = false;
  startSpeed = DEC_START_SPEED;
  targetSpeed = DEC_TARGET_SPEED;
}

void startMotor() {
  digitalWrite(MOTOR_IN1, HIGH);
  digitalWrite(MOTOR_IN2, LOW);

  analogWrite(MOTOR_ENA, startSpeed);
  digitalWrite(STATUS_LED, HIGH);

  motorRunning = true;
  isRamping = true;
  rampStartTime = millis();
}

void handleRamping() {
  if (!motorRunning || !isRamping) return;

  unsigned long elapsed = millis() - rampStartTime;

  if (elapsed < RAMP_DURATION) {
    int speed = map(elapsed, 0, RAMP_DURATION, startSpeed, targetSpeed);
    analogWrite(MOTOR_ENA, speed);
  } else {
    analogWrite(MOTOR_ENA, targetSpeed);
    isRamping = false;
  }
}

void stopMotor() {
  digitalWrite(MOTOR_IN1, LOW);
  digitalWrite(MOTOR_IN2, LOW);
  analogWrite(MOTOR_ENA, 0);

  digitalWrite(STATUS_LED, LOW);

  motorRunning = false;
  isRamping = false;
}
