import os
import time

# log function (logging does not work with wafari)
def addlog(flog,src,message,type="INFO",line=-1):
    """ Add a log line to the flog file """
    with open(flog,'a') as logfile:
        txt =  "%s - %s:%s:%s - %s\n"%(time.strftime("%Y-%m-%d %H:%M"),
                                    src,line,type,message)
        logfile.write(txt)

# end of log function
