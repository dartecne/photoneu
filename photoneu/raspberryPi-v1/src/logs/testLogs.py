import logging
import time
import random

log_info = logging.getLogger("info")
log_data = logging.getLogger("data")
log_info.setLevel(logging.INFO)
log_data.setLevel(logging.INFO)
d = {'tns':time.clock_gettime_ns(0)}
formater= logging.Formatter('%(message)s')
info_fh = logging.FileHandler('info.log')
data_fh = logging.FileHandler('data.log')
info_fh.setFormatter( formater )
data_fh.setFormatter( formater )
log_info.addHandler( info_fh )
log_data.addHandler( data_fh )


def main():
    x = random.random()
    msg="t1,x1,t2,x2"
    log_data.info( msg )
    for i in range(12):
        ts = time.clock_gettime_ns(0)
        msg = str(ts) + "," + str(x) 
        time.sleep(0.5)
        x += random.random()
        ts = time.clock_gettime_ns(0)
        msg += "," + str(ts) + "," + str(x)
        log_data.info( msg )
#        ts = time.gmtime()

if __name__ == '__main__':
    main()