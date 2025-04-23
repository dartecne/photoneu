from __future__ import print_function
import argparse
import time

from motorHandler import MotorHandler

motor = MotorHandler()
time.sleep( 2 )
print("position:")
print( motor.getMotorPosition() )
t, x, y = motor.getSPerror()

parser = argparse.ArgumentParser()
  
def main( x, y ):
    motor.moveHead(int(x),y)


if __name__ == '__main__':
     # Adding optional argument
    parser.add_argument("-x", "--point_x", help = "Move to XY point in number of steps", type = int)
    parser.add_argument("-y", "--point_y", help = "Move to XY point in number of steps", type = int)
    args = parser.parse_args()
    main( args.point_x, args.point_y )