import time
import serial

# configure the serial connections (the parameters differs on the device you are connecting to)
ser = serial.Serial(
    port='/dev/ttyACM0',
    baudrate=115200,
    parity=serial.PARITY_ODD,
    stopbits=serial.STOPBITS_TWO,
    bytesize=serial.SEVENBITS
)

ser.isOpen()

print ('Enter your commands below.\r\nInsert "exit" to leave the application.')
msg = ''
#input=1
print (ser.readline().decode())
print (ser.readline().decode())

while 1 :
    # get keyboard input
#    input = raw_input(">> ")
        # Python 3 users
    msg = input(">> ")
    if msg == 'exit':
        ser.close()
        exit()
    else:
        # send the character to the device
        # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
        ser.write(msg.encode())
        ser.write('\n'.encode())
        out = ''
        # let's wait one second before reading output (let's give device time to answer)
        while (ser.inWaiting() ):
        #    out += ser.read(1).decode()
            print(ser.readline().decode())
            print(ser.readline().decode())
        if out != '':
            print (">>" + out)
