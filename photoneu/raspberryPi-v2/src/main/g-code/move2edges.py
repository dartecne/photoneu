import time
import serial

# Colocar el cabezal en la posicion XY: 0, 0
# El programa movera el cabezal por los limites del plano y terminara en el centroxmai

ymax = 6
port = '/dev/ttyACM0'
#port = '/dev/ttyACM1'
baudrate = 115200
# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial( port, baudrate )
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
        print (ser.readline().decode())
        
def testEdges() :
    sendCode( 'G0 X0Y0Z0' )
    time.sleep(0.5)
    sendCode( 'G0 X-4Y0Z0' )
    time.sleep(0.5)
    sendCode( 'G0 X-4Y6Z6' )
    time.sleep(0.5)
    sendCode( 'G0 X0Y6Z6' )
    time.sleep(0.5)
    sendCode( 'G0 X0Y0Z0' )
    time.sleep(0.5)

#msg = 'G01 Y1Z1 F60'.encode()
#msg = 'G01 X2 F40'.encode()
#msg = 'M17'.encode()
#msg = '$H'.encode()
msg = 'G01 Y3Z3 F1000'
sendCode( '$' )
sendCode( '$X' )
sendCode( 'G90' ) # coordenadas absolutas
#sendCode( '$X' )
#sendCode( 'G28' )
#sendCode( '$H' )
testEdges()
time.sleep(1)

print(">>>exiting")

ser.close()
exit()
