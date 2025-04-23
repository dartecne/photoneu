from __future__ import print_function
import argparse
import time

from motorHandler import MotorHandler

motor = MotorHandler()
time.sleep( 2 )
print("position:")
print( motor.getMotorPosition() )
t, x, y = motor.getSPerror()

motor.moveHead(1000,1000)
t, x, y = motor.getSPerror()
while ( x != 0 ) | ( y != 0 ):
    t, x, y = motor.getSPerror()
    print("position = ")
    print(motor.getMotorPosition())
    print("error = ") 
    print( motor.getSPerror())

motor.moveHead(10000,10000)
t, x, y = motor.getSPerror()
while ( x != 0 ) | ( y != 0 ):
    t, x, y = motor.getSPerror()
    print("position = ")
    print(motor.getMotorPosition())
    print("error = ") 
    print( motor.getSPerror())
