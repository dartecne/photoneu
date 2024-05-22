import logging
import time
import random

log_info = logging.getLogger("info")
log_data = logging.getLogger("data")
log_info.setLevel(logging.INFO)
log_data.setLevel(logging.INFO)
d = {'tns':time.clock_gettime_ns(0)}
formater= logging.Formatter('%(tns)s,%(message)s')
info_fh = logging.FileHandler('info.log')
data_fh = logging.FileHandler('data.log')
info_fh.setFormatter( formater )
data_fh.setFormatter( formater )
log_info.addHandler( info_fh )
log_data.addHandler( data_fh )


def main():
    x = random.random()
    for i in range(12):
        x += random.random()
        ts = time.clock_gettime_ns(0)
        log_info.info(str(i) +"," + str(x), extra=d)
        log_data.info(str(i), extra=d)
        time.sleep(0.5)
#        ts = time.gmtime()

if __name__ == '__main__':
    main()