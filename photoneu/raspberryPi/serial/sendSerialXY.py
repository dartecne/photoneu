import time
import serial

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='/dev/ttyACM0',
    baudrate=230400,
    #baudrate=115200,
    #baudrate=9600,
    #parity=serial.PARITY_ODD,
    #parity=serial.PARITY_EVEN,
    #stopbits=serial.STOPBITS_TWO,
    #stopbits=serial.STOPBITS_ONE,
    #bytesize=serial.SEVENBITS
)

ser.isOpen()

print ('Enter your commands below.\r\nInsert "exit" to leave the application.')
msg = "X10000Y12000\r\n"
#input=1
print( "Sending... " + msg );
        #ser.write(msg.encode())
ser.write(bytes(msg, 'utf-8'))
   #     ser.write('\r\n'.encode())
        # let's wait one second before reading output (let's give device time to answer)
time.sleep(1)
while (ser.inWaiting() ):
    print(ser.readline().decode())

while 1 :
    # get keyboard input
#    input = raw_input(">> ")
        # Python 3 users
    msg = input(">> ")
#    msg = "X0040Y0060"
    if msg == 'exit':
        ser.close()
        exit()
    else:
        # send the character to the device
        # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
        print( "Sending... " + msg );
        ser.write(msg.encode(encoding='ascii'))
        #ser.write(bytes(msg, 'utf-8'))
   #     ser.write('\r\n'.encode())
        # let's wait one second before reading output (let's give device time to answer)
        while (ser.inWaiting() ):
            print(ser.readline().decode())
