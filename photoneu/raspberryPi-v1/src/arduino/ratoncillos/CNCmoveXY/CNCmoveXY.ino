/**
  Recibe datos por el puerto serie con el formato: XnnnnYmmmm
  Ejemplo: X0123Y2040
  que corresponden al número de steps que tienen que hacer los motores para moverse en el plano XY
*/


const int enPin=8;
const int stepXPin = 2; //X.STEP
const int dirXPin = 5; // X.DIR
const int stepYPin = 3; //Y.STEP
const int dirYPin = 6; // Y.DIR
const int stepZPin = 4; //Z.STEP
const int dirZPin = 7; // Z.DIR
bool xDir;
bool yDir;

const int limitXPin = 9;
const int limitYPin = 10;
const int limitZPin = 11;

const int stepsPerRev=200;
unsigned long pulseWidthMicros = 10; 	// microseconds
unsigned long microsBtwnSteps = 100; // milliseconds
unsigned long maxMicrosBtwnSteps= 200;
unsigned long minMicrosBtwnSteps= 20;

unsigned long curMicros;
unsigned long prevStepMicros = 0;

int xPos = 0, yPos = 0;
//int yMax = 1540;  // valores aproximados obtenidos después de la calibración
int yMax = 24272;  // resolucion de 1/16
//int xMax = 1250;  
int xMax = 19430;  // 19431 con una resolucion de 1/16
int xSP = 0, ySP = 0; // Set Point, donde debe ir el cabezal
bool stop = true;

void setup() {
 	Serial.begin(115200);
  while( !Serial );
 	pinMode(enPin, OUTPUT);
 	digitalWrite(enPin, LOW);

 	pinMode(stepXPin, OUTPUT);
 	pinMode(stepYPin, OUTPUT);
 	pinMode(stepZPin, OUTPUT);

 	pinMode(dirXPin, OUTPUT);
 	pinMode(dirYPin, OUTPUT);
 	pinMode(dirZPin, OUTPUT);

  pinMode(limitXPin, INPUT_PULLUP);
  pinMode(limitYPin, INPUT_PULLUP);
  pinMode(limitZPin, INPUT_PULLUP);
 
 	digitalWrite(dirXPin, HIGH); // Enables the motor to move in a particular direction
 	digitalWrite(dirYPin, HIGH); // Enables the motor to move in a particular direction
 	digitalWrite(dirZPin, HIGH); // Enables the motor to move in a particular direction
//  testX();
//  testYZ();
//  microsBtwnSteps = minMicrosBtwnSteps;
  microsBtwnSteps = maxMicrosBtwnSteps;
 	Serial.println("CNC Shield Initialized");
  calibrate();
  setPoint( xMax/2, yMax/2 );
  stop = false;

}

void loop() {
  curMicros = micros();
  while( stop ) checkLimits();
  if(xSP - xPos > 0 ) setXdirection(HIGH); else setXdirection(LOW);
  if(ySP - yPos > 0 ) setYdirection(HIGH); else setYdirection(LOW);
  if( xSP != xPos ) singleStep( stepXPin );
  if( ySP != yPos ) {
   singleStep( stepYPin );
   singleStep( stepZPin );
  }
//  if( xSP == xPos & ySP == yPos ) Serial.println( "DONE!" );;
  delayMicroseconds(microsBtwnSteps);  
  if(readSerialData() > 0) setPoint(xSP, ySP);
}

int readSerialData() {
  if(!Serial.available()) return 0; // espera datos
  String str = "X1234Y1234";
//  str = Serial.readString();
  str = Serial.readStringUntil('\0');
  str.trim();
  Serial.print("Serial Received: ");Serial.println(str);
  Serial.print("length: "); Serial.println(str.length());
//  if(str.length() != 10 ) return;
 // printStr(str);
  Serial.println(  "Getting values..." );
  if(str[0] == 'X') xSP = string2number( str, 1, 5 );
//  Serial.println( xSP );
  if(str[6] == 'Y') ySP = string2number( str, 7, 5 );
  return str.length();
}

void printStr( String str ) {
  for( int i = 0; i < str.length(); i++ ) {
    Serial.print( str[i] ); Serial.print( " " ); Serial.println( str[i], DEC );
//    Serial.write(str[i]);
  }
}

int string2number(String str, int init, int num) {
  String strTemp = "";
  for( int i = 0; i < num; i++ ) {
    strTemp += str.charAt(init + i);
 //   Serial.println(strTemp);
  }
  return(strTemp.toInt());
}

/**
* Busca el (0,0)
*/
void calibrate() {
  Serial.println("Calibrating...");
  setYdirection(HIGH);
  while( digitalRead(limitYPin) ) {
    singleStep( stepYPin );
    singleStep( stepZPin );
    delayMicroseconds(microsBtwnSteps);
  }
  setYdirection(LOW);
  while( !digitalRead(limitYPin) ) {
    singleStep( stepYPin );
    singleStep( stepZPin );
    delayMicroseconds(microsBtwnSteps);
  }
  Serial.println("Y calibrated!");
  Serial.print("Y steps: ");Serial.println( yPos );
  yPos = yMax;
  setXdirection(HIGH);
  while( digitalRead(limitXPin) ) {
    singleStep( stepXPin );
    delayMicroseconds(microsBtwnSteps);
  }
  setXdirection(LOW);
  while( !digitalRead(limitXPin) ) {
    singleStep( stepXPin );
    delayMicroseconds(microsBtwnSteps);
  }
  Serial.println("X calibrated!");
  Serial.print("X steps: ");Serial.println( xPos );
  xPos = xMax;
}

void setPoint( unsigned int x, unsigned int y ) {
  xSP = x;
  ySP = y;
  stop = false;
  Serial.print( "Going to xSP: " ); Serial.print(xSP); 
  Serial.print(" ySP: "); Serial.println( ySP );
}

void setDirection( bool xD, bool yD ) {
  setXdirection( xD );
  setYdirection( yD );
}

void setXdirection( bool xD ) {
  digitalWrite(dirXPin, xD); // Enables the motor to move in a particular direction
  xDir = xD;
}

void setYdirection( bool yD ) {
  digitalWrite(dirYPin, yD); // Enables the motor to move in a particular direction
  digitalWrite(dirZPin, yD); // Enables the motor to move in a particular direction
  yDir = yD;
}

void singleStep( int sPin) {
//  if (curMicros - prevStepMicros > microsBtwnSteps) {
//        prevStepMicros = curMicros;
  digitalWrite(sPin, HIGH);
  delayMicroseconds(pulseWidthMicros);
  digitalWrite(sPin, LOW);
  if(sPin == stepXPin) {
    if(xDir) xPos++; else xPos--;
  }
  else if(sPin == stepYPin) {
    if(yDir) yPos++; else yPos--;
  }
  if(xPos < 0 ) xPos = 0;
  if(yPos < 0 ) yPos = 0;
//  Serial.print(xPos); Serial.print(", "); Serial.println(yPos);

  checkLimits();
}

void checkLimits() {
  if( !digitalRead(limitXPin) ) Serial.println("Limit X!");
  if( !digitalRead(limitYPin) ) Serial.println("Limit Y!");
}

void testYZ() {
 	Serial.println(F("Running clockwise"));
 	digitalWrite(dirYPin, HIGH); // Enables the motor to move in a particular direction
 	digitalWrite(dirZPin, HIGH); // Enables the motor to move in a particular direction
 	// Makes 200 pulses for making one full cycle rotation
 	for (int i = 0; i < stepsPerRev; i++) {
 			digitalWrite(stepYPin, HIGH);
 			digitalWrite(stepZPin, HIGH);
 			delayMicroseconds(pulseWidthMicros);
 			digitalWrite(stepYPin, LOW);
 			digitalWrite(stepZPin, LOW);
 			delayMicroseconds(microsBtwnSteps);
 	}
 	delay(1000); // One second delay

 	Serial.println(F("Running counter-clockwise"));
 	digitalWrite(dirYPin, LOW); //Changes the rotations direction
 	digitalWrite(dirZPin, LOW); //Changes the rotations direction
 	// Makes 400 pulses for making two full cycle rotation
 	for (int i = 0; i < stepsPerRev; i++) {
 			digitalWrite(stepYPin, HIGH);
 			digitalWrite(stepZPin, HIGH);
 			delayMicroseconds(pulseWidthMicros);
 			digitalWrite(stepYPin, LOW);
 			digitalWrite(stepZPin, LOW);
 			delayMicroseconds(microsBtwnSteps);
 	}
 	delay(1000);
}

void testX() {
 	Serial.println(F("Running clockwise"));
 	digitalWrite(dirXPin, HIGH); // Enables the motor to move in a particular direction
 	// Makes 200 pulses for making one full cycle rotation
 	for (int i = 0; i < stepsPerRev; i++) {
 			digitalWrite(stepXPin, HIGH);
 			delayMicroseconds(pulseWidthMicros);
 			digitalWrite(stepXPin, LOW);
 			delayMicroseconds(microsBtwnSteps);
 	}
 	delay(1000); // One second delay

 	Serial.println(F("Running counter-clockwise"));
 	digitalWrite(dirXPin, LOW); //Changes the rotations direction
 	// Makes 400 pulses for making two full cycle rotation
 	for (int i = 0; i < stepsPerRev; i++) {
 			digitalWrite(stepXPin, HIGH);
 			delayMicroseconds(pulseWidthMicros);
 			digitalWrite(stepXPin, LOW);
 			delayMicroseconds(microsBtwnSteps);
 	}
 	delay(1000);
}
