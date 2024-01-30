/* Example sketch to control a stepper motor with A4988 stepper motor driver 
   and Arduino without a library. More info: https://www.makerguides.com */

// Define stepper motor connections and steps per revolution:
// Utilizar el cable de conexión estandar para el stepper tipo 17HS4401 o 17HD0004-05, es dedir los que tienes 6 pines.
// RED - NC - GREEN - BLUE - NC - BLACK
// RED - 2B
// BLUE - 2A
// GREEN - 1A
// BLACK - 1B

// para los NEMA de 4 pines con el cable ya incluído no hay que hacer nada especial, y el orden de colores es idéntico.


// en el A4988 conectar pin SLEEP con RESET. En algunos foros se comenta también llevarlo a VCC
// 
#define dirPin 2
#define stepPin 3
#define stepsPerRevolution 200

int d = 1000;
void setup() {
  // Declare pins as output:
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
}

void loop() {
  // Set the spinning direction clockwise:
  digitalWrite(dirPin, HIGH);

  // Spin the stepper motor 1 revolution slowly:
  for (int i = 0; i < stepsPerRevolution; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(d);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(d);
  }

  delay(1000);

  // Set the spinning direction counterclockwise:
  digitalWrite(dirPin, LOW);

  // Spin the stepper motor 1 revolution quickly:
  for (int i = 0; i < stepsPerRevolution; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(d);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(d);
  }

  delay(1000);

/*  // Set the spinning direction clockwise:
  digitalWrite(dirPin, HIGH);

  // Spin the stepper motor 5 revolutions fast:
  for (int i = 0; i < 5 * stepsPerRevolution; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }

  delay(1000);

  // Set the spinning direction counterclockwise:
  digitalWrite(dirPin, LOW);

  //Spin the stepper motor 5 revolutions fast:
  for (int i = 0; i < 5 * stepsPerRevolution; i++) {
    // These four lines result in 1 step:
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }

  delay(1000);
  */
}