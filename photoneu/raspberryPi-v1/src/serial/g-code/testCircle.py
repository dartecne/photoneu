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

#while ser.inWaiting() > 0:
#print (ser.readline(ser.inWaiting()))
print(">>>serial says: ")
print (ser.readline().decode())
while(ser.inWaiting()) :
    print (ser.readline().decode())
         
msg = 'G01 Y1Z1 F60'.encode()
#msg = 'G01 X2 F40'.encode()
#msg = 'M17'.encode()
#msg = '$'.encode()
print(">>>writing... ")
ser.write( msg );
ser.write('\n'.encode())
print(">>>serial says: ")
print (ser.readline().decode())
#time.sleep(3)

msg = 'G01 Y0Z0 F60'.encode()
print(">>>writing... ")
ser.write( msg );
ser.write('\n'.encode())
print(">>>serial says: ")
print (ser.readline().decode())

#time.sleep(1)
while(ser.inWaiting()) :
    print (ser.readline().decode())
time.sleep(1)
print(">>>exiting")

ser.close()
exit()
