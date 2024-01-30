import time
import serial
port = '/dev/ttyACM0'
#port = '/dev/ttyACM1'
baudrate = 115200
# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial( port, baudrate )
#    parity=serial.PARITY_ODD,
#    stopbits=serial.STOPBITS_TWO,
#    bytesize=serial.SEVENBITS
#)
ser.setDTR(0)
if ser.isOpen(): print(">>>Opened port: \"%s\"." % (port))
out = ''

ser.write("\r\n\r\n".encode());
time.sleep(2)
#ser.flushInput() 

def sendCode( msg ) :
    print(">>>writing... ")
    ser.write( msg.encode() );
    ser.write('\n'.encode())
    print(">>>serial says: ")
    while ser.inWaiting() > 0:
#        print (ser.readline(ser.inWaiting()))
        print (ser.readline().decode())
         
#msg = 'G01 Y1Z1 F60'.encode()
#msg = 'G01 X2 F40'.encode()
#msg = 'M17'.encode()
#msg = '$H'.encode()
msg = 'G01 Y3Z3 F1000'
sendCode( '$' )
sendCode( '$X' )
sendCode( 'G0 X3' )
sendCode( 'G28' )
#sendCode( '$H' )
time.sleep(1)

print(">>>exiting")

ser.close()
exit()
