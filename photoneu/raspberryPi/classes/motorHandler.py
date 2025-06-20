from __future__ import print_function
import numpy as np
import serial

class MotorHandler:
    def __init__(self):
        self.port = '/dev/ttyACM0'
        self.baudrate = 230400
        self.x = 0
        self.x_max = 22000
        self.y = 0
        self.y_max = 24000
        while True:
            try:
                self.ser = serial.Serial( self.port, self.baudrate )
                break
            except:
                print("Unable to open Serial Port: %s" % (self.port))
                print("Trying Port: /dev/ttyACM1")
                self.port = '/dev/ttyACM1'
        if self.ser.isOpen(): print(">>>Opened port: \"%s\"." % (self.port))
        else:
                print("Unable to open Serial Port: %s" % (self.port))
                print(">>>exiting")
                exit()

    def endSystem( self ):
        print( "MotorHandler::exiting" )
        serial.Serial.close()
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

    def pbmOn(self):
        msg="H\0"
        self.ser.write( msg.encode(encoding= 'ascii') )

    def pbmOff(self):
        msg="L\0"
        self.ser.write( msg.encode(encoding= 'ascii') )

    def moveHead( self, x_head, y_head ):
        x_head = max(0,x_head)
        y_head = max(0,y_head)
        x_head = min(self.x_max,x_head)
        y_head = min(self.y_max,y_head)
        print( "motorHandler::moving to: " + str(x_head) + str(", ") + str(y_head) )
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
        self.x = int(x_head)
        self.y = int(y_head)
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

    def printValues( self, t, x_head, y_head ):
#        t, x_head, y_head = self.getMotorPosition()
        line = str(t) + \
            "," + str(x_head) + "," + str(y_head)
        print( "head, = " + line )
        t, x_head_error, y_head_error = self.getSPerror()
        print( "head_error = " + str(t) + "," + str(x_head_error) + "," + str(y_head_error) )
