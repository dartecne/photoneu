const int enPin=8;
const int stepXPin = 2; //X.STEP
const int dirXPin = 5; // X.DIR
const int stepYPin = 3; //Y.STEP
const int dirYPin = 6; // Y.DIR
const int stepZPin = 4; //Z.STEP
const int dirZPin = 7; // Z.DIR

int stepPin=stepXPin;
int dirPin=dirXPin;

const int stepsPerRev=200;
unsigned long pulseWidthMicros = 0; 	// microseconds
unsigned long microsBtwnSteps = 2000; // milliseconds

unsigned long curMicros;
unsigned long prevStepMicros = 0;

void setup() {
 	Serial.begin(9600);
 	pinMode(enPin, OUTPUT);
 	digitalWrite(enPin, LOW);
 	pinMode(stepPin, OUTPUT);
 	pinMode(dirPin, OUTPUT);
 
 	Serial.println(F("CNC Shield Initialized"));
 	digitalWrite(dirPin, HIGH); // Enables the motor to move in a particular direction
  test();
}

void loop() {
  curMicros = micros();
//  for( int i = 0; i < stepsPerRev; i++ ) {    
  if (curMicros - prevStepMicros > microsBtwnSteps) {
        prevStepMicros = curMicros;

      singleStep();
//    delay(millisBtwnSteps);
  }
}

void test() {
 	Serial.println(F("Running clockwise"));
 	digitalWrite(dirPin, HIGH); // Enables the motor to move in a particular direction
 	// Makes 200 pulses for making one full cycle rotation
 	for (int i = 0; i < stepsPerRev; i++) {
 			digitalWrite(stepPin, HIGH);
 			delayMicroseconds(pulseWidthMicros);
 			digitalWrite(stepPin, LOW);
 			delayMicroseconds(microsBtwnSteps);
 	}
 	delay(1000); // One second delay

 	Serial.println(F("Running counter-clockwise"));
 	digitalWrite(dirPin, LOW); //Changes the rotations direction
 	// Makes 400 pulses for making two full cycle rotation
 	for (int i = 0; i < stepsPerRev; i++) {
 			digitalWrite(stepPin, HIGH);
 			delayMicroseconds(pulseWidthMicros);
 			digitalWrite(stepPin, LOW);
 			delayMicroseconds(microsBtwnSteps);
 	}
 	delay(1000);
}


void singleStep() {
//    if (curMillis - prevStepMillis >= millisBtwnSteps) {
            // next 2 lines changed 28 Nov 2018
        //prevStepMillis += millisBetweenSteps;
        digitalWrite(stepPin, HIGH);
  			delayMicroseconds(pulseWidthMicros);
        digitalWrite(stepPin, LOW);
//    }
}
