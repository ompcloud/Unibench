#!/usr/bin/env python3

import sys
import re

class OmpcloudException(Exception):
    pass

def extract_values(str):
    return re.findall(r"[-+]?\d*\.\d+|\d+", str)

def analyze_spark_log(filepath, verbose=True):
    logfile = open(filepath)
    total_comp_time = 0.0

    for line in logfile:
        if "SparkException" in line:
            raise OmpcloudException()

        if line.startswith("Target Cloud RTL --> Execution = "): #you might want a better check here
            str_spark_time = line.split(" = ",1)[1]
            spark_time = float(extract_values(str_spark_time)[0])
            if verbose:
                print("Spark time = " + str(spark_time))

        if bool(re.search('scheduler.DAGScheduler: Job [0-9]+ finished', line)):
            str_comp_time = line.split(" took ",1)[1]
            comp_time = float(extract_values(str_comp_time)[0])
            total_comp_time += comp_time
            if verbose:
                print("Computation time = " + str(comp_time))

        if line.startswith("GPU Runtime: "): #you might want a better check here
            str_full_time = line.split(": ",1)[1]
            full_time = float(extract_values(str_full_time)[0])
            if verbose:
                print("Full time = " + str(full_time))

    if verbose:
        print("{:.1f},{:.1f},{:.1f}".format(full_time,spark_time,total_comp_time))

    return [full_time,spark_time,total_comp_time]



def analyze_output_log(filepath, verbose=True):
    dic_results = {}
    logfile = open(filepath)

    line = logfile.readline()
    while line.startswith("Run benchmarks on "):
        nbcore = int(extract_values(line)[0])
        if verbose:
            print("Nbcore " + str(nbcore))

        line = logfile.readline()
        while line.startswith("-- Suite "):
            suite = line.split("-- Suite ",1)[1].rstrip('\n')
            #print(" - " + suite)

            line = logfile.readline()
            while line.startswith("--- Benchmark"):
                bench = line.split("--- Benchmark ",1)[1].split(":",1)[0]

                execution_times = []
                line = logfile.readline()
                while line.startswith("Execution "):
                    execution_times.append(extract_values(line)[1])
                    line = logfile.readline()

                if not execution_times:
                    continue

                if line.startswith("Median:"):
                    # line contains Median is only printed in recent log
                    line = logfile.readline()
                # line contains Variance
                line = logfile.readline()
                # line contains Std deviation

                min_execution = min(execution_times)
                index = execution_times.index(min_execution)

                if verbose:
                    print(" -- " + bench + " n" + str(index) + " = " + str(min_execution))
                bench_logfile = "log-{}cores-{}_{}.{}.out".format(nbcore,suite,bench,index)

                try:
                    results = analyze_spark_log(bench_logfile, verbose)
                    dic_results.setdefault(bench,[]).append([nbcore, results])
                except OmpcloudException as e:
                    print("Warning: Failed for " + bench + " on " + str(nbcore) + " cores")
                    pass

                line = logfile.readline()

    return dic_results

def pretty_printing(results):
    for bench, resultsByNbcore in results.items():
        print(bench)
        for e in resultsByNbcore:
            csv_line = "{},{:.1f},{:.1f},{:.1f}".format(e[0],e[1][0],e[1][1],e[1][2])
            print(csv_line)


if len(sys.argv) != 2:
    RuntimeException("wrong arguments", false)

print("Analyzing " + sys.argv[1] + " .... ")

parsed_results = analyze_output_log(sys.argv[1],False)

pretty_printing(parsed_results)
