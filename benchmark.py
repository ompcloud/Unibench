#!/usr/bin/env python3

import subprocess
import timeit
import os.path
import numpy
import time
import sys

NB_EXEC = 2 # Number of time the execution is repeated
UNIBENCH_BUILD = "/opt/Unibench-build/" # Path to the benchmarks
LOG_DIR = "/opt/Unibench-log/"
#NB_CORE = [ 8, 16, 32, 64, 128, 256 ]
NB_CORE = [ 8 ]
APPLICATION = [
    ("mgBench", ["mat-mul", "mat-sum"])
]
OMPCLOUD_CONF = "/opt/ompcloud-conf/cloud_rtl.ini.aws_c3.8xlarge"

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if not os.path.exists(UNIBENCH_BUILD) :
    raise Exception("Unibench build directory does not exist: " + UNIBENCH_BUILD)
if not os.path.exists(LOG_DIR) :
    raise Exception("Log directory does not exist: " + LOG_DIR)

timestr = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(LOG_DIR, timestr)
os.mkdir(log_dir)
# Write message into a file as well
sys.stdout = Logger(os.path.join(log_dir, "output.log"))

for nb_core in NB_CORE :
    print("Run benchmarks on " + str(nb_core) + " worker cores")
    #os.environ['OMPCLOUD_CONF_PATH'] = OMPCLOUD_CONF + "." + str(nb_core)
    for (suite, benchmarks) in APPLICATION :
        print("-- Suite " + suite)
        for bench in benchmarks :
            binary = os.path.join(UNIBENCH_BUILD, "benchmarks", suite, bench, bench)
            if not os.path.exists(binary) :
                raise Exception("Binary does not exist: " + binary)
            cmd = os.path.abspath(binary)
            print("--- Benchmark " + bench + ": " + binary)
            times = []
            for n in range(0,NB_EXEC) :
                log = "log-{}cores-{}_{}.{}.out".format(nb_core,suite,bench,n)
                logfile = open(os.path.join(log_dir, log), "w")

                start = timeit.default_timer()
                subprocess.call([cmd], stdout=logfile, stderr=subprocess.STDOUT)
                elapsed = timeit.default_timer() - start

                print("Execution {} in {:.2f}s".format(n, elapsed))
                logfile.close()
                times.append(elapsed)
                time.sleep(2) # wait to avoid JVM exception

            print("Variance: " + str(numpy.var(times)))
            print("Std deviation: " + str(numpy.std(times)))
