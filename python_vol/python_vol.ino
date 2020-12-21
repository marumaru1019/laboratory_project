#include <MsTimer2.h>
unsigned long time;
unsigned int val0;

double value;
char x;
const int led = 5;

const int PIN_PWM = 13;
const float V_OUT = 2;




void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  value = 0.0;
}
void loop() {
  // put your main code here, to run repeatedly:
//  1あたり20mV
  if (Serial.available()) {
    x = Serial.read();

    if (x == ' ') {
      time = micros();
      value = analogRead(A0);
      value *=  5.0 / 1024;
      Serial.print(time);
      Serial.print(" ");
      Serial.println(value);
    }
  }
}
