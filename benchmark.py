#!/usr/bin/env python3

import subprocess
import timeit
import os.path
import numpy
import time
import sys
import atexit

WORKING_DIR = "/opt"

NB_EXEC = 3 # Number of time the execution is repeated
NB_CORE = [ 256, 192, 128, 64, 32, 16, 8 ]
#NB_CORE = [ 32, 16 ]
#NB_CORE = [ 1 ]
APPLICATION = [
#    ("mgBench", ["mat-mul", "collinear-list"])
    ("mgBench", ["mat-mul", "collinear-list"]),
    ("Polybench", ["2MM", "3MM", "COVAR", "GEMM", "SYR2K", "SYRK"])
]

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

#@atexit.register
def shutdown_aws_cluster():
    subprocess.run(["cgcloud", "terminate-cluster", "-c", "ompcloud-experiments", "spark"], check=True)

UNIBENCH_BUILD = os.path.join(WORKING_DIR, "Unibench-build") # Path to the benchmarks
LOG_DIR = os.path.join(WORKING_DIR, "Unibench-log")
OMPCLOUD_CONF = os.path.join(WORKING_DIR, "ompcloud-conf")
OMPCLOUD_CONF_PREFIX = "cloud_rtl.ini.aws_iccp"

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
    conf_filename = OMPCLOUD_CONF_PREFIX + "." + str(nb_core)
    conf_filepath = os.path.join(OMPCLOUD_CONF, conf_file)
    if not os.path.exists(conf_filepath) :
        print("Warning - Configuration file does not exist: " + conf_filepath)
    os.environ['OMPCLOUD_CONF_PATH'] = conf_filepath
    for (suite, benchmarks) in APPLICATION :
        print("-- Suite " + suite)
        for bench in benchmarks :
            bench_dir = os.path.join(UNIBENCH_BUILD, "benchmarks", suite, bench)
            if not os.path.exists(bench_dir) :
                raise Exception("Benchmark directory do not exist: " + bench_dir)
            os.chdir(bench_dir)
            binary = os.path.abspath(bench)
            if not os.path.exists(binary) :
                raise Exception("Binary does not exist: " + binary)
            print("--- Benchmark " + bench + ": " + binary)
            times = []
            for n in range(0,NB_EXEC) :
                log = "log-{}cores-{}_{}.{}.out".format(nb_core,suite,bench,n)
                logfile = open(os.path.join(log_dir, log), "w")

                start = timeit.default_timer()
                subprocess.run([binary], stdout=logfile, stderr=logfile, check=True)
                elapsed = timeit.default_timer() - start

                print("Execution {} in {:.2f}s".format(n, elapsed))
                logfile.close()
                times.append(elapsed)
                time.sleep(10) # wait to avoid JVM exception

            print("Variance: " + str(numpy.var(times)))
            print("Std deviation: " + str(numpy.std(times)))
