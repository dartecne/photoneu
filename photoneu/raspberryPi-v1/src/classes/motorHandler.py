from __future__ import print_function
import argparse
import numpy as np
import time
import serial

class MotorHandler:
    def __init__(self):
        self.port = '/dev/ttyACM0'
        self.baudrate = 230400
        self.ser = serial.Serial( self.port, self.baudrate )
        if self.ser.isOpen(): print(">>>Opened port: \"%s\"." % (self.port))
        else :
            print("Unable to open Serial Port: %s" % (self.port))
            print(">>>exiting")
            exit()

    def sendCode( self, msg ) :
        print(">>> sendCode ")
        msg += "\0"
        self.ser.write( msg.encode(encoding= 'ascii') )
#    while ser.inWaiting() > 0:
#        print (".")
#        print ("<<<" + ser.readline().decode() )

    def sendCalibrate(self) :
        msg = "C\0"
        self.ser.write( msg.encode(encoding= 'ascii') )    

    def moveHead( self, x_head, y_head ):
        print( "moving to: " + str(x_head) + str(", ") + str(y_head) )
        msg = "X" + str(int(x_head)).zfill(5) + "Y" + str(int(y_head)).zfill(5) 
        self.sendCode(msg)
    
    def getMotorPosition(self):   
        msg = "P\0"
        self.ser.write( msg.encode(encoding= 'ascii') )
        line = self.ser.readline().decode('utf-8','ignore').rstrip() #replace, backslashreplace
    #    print("<<< reading... ")
    #    print(line)
    #    print(len(line.split(",")))
        timestamp, x_head, y_head = -1,-1,-1
        if len(line.split(",")) == 3:
            timestamp, x_head, y_head = line.split(",")
        if(timestamp == -1 ): 
            print("ERROR motor position")
            print("<<< reading... ")
            print(line)
        return int(timestamp), int(x_head), int(y_head)

    def getSPerror(self) :
        msg = "E\0"
        self.ser.write( msg.encode(encoding= 'ascii') )    
        line = self.ser.readline().decode('utf-8','ignore').rstrip()
    #    print("<<< reading... ")
    #    print(line)
    #    print(len(line.split(",")))
        timestamp, x_head_error, y_head_error = -1,-1,-1
        if len(line.split(",")) == 3:
            timestamp, x_head_error, y_head_error = line.split(",")
        if(timestamp == -1 ): 
            print("ERROR lectura SPerror")
            print("<<< reading... ")
            print(line)
        return int(timestamp), int(x_head_error), int(y_head_error)
