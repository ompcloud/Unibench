#!/usr/bin/env python3

"""
A script to generate bar diagrams with benchmarks results
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import os.path
import sys

APPLICATION = [
    ("mgBench", ["mat-mul", "collinear-list"]),
    ("Polybench", ["2MM", "3MM", "COVAR", "GEMM", "SYR2K", "SYRK"])
]

N = 7

#Nbcore = [8, 16, 32, 64, 128, 192, 256]
nbcore2ind = dict([("1", -1), ("8", 0), ("16", 1), ("32", 2), ("64", 3), ("128", 4), ("192", 5), ("256", 6)])

def GenGraphes(bench, path):

    ompthread = np.zeros(N)
    ompcloud_comp_all = np.zeros([N,2])
    ompcloud_spark_all = np.zeros([N,2])
    ompcloud_full_all = np.zeros([N,2])

    csvFile = os.path.join(path, "bench_" + bench + ".csv")

    with open(csvFile) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            i = nbcore2ind[row['nbcore']]
            if row['type'] == "sequential":
                sequential = float(row['full_time'])
            elif row['type'] == "ompthread":
                ompthread[i] = float(row['full_time'])
            elif row['type'].startswith("ompcloud-"):
                j = 0
                if row['type'] == "ompcloud-full":
                    j = 1
                ompcloud_comp_all[i][j] = float(row['comp_time'])
                ompcloud_spark_all[i][j] = float(row['spark_time'])
                ompcloud_full_all[i][j] = float(row['full_time'])
            else:
                raise RuntimeException("Unexpected CSV format")


    # Compute speedup
    sp_ompthread = np.divide(sequential, ompthread)
    sp_ompcloud_comp_all = np.divide(sequential, ompcloud_comp_all)
    sp_ompcloud_spark_all = np.divide(sequential, ompcloud_spark_all)
    sp_ompcloud_full_all = np.divide(sequential, ompcloud_full_all)

    # Compute mean and deviation for full and sparse matrices
    sp_ompcloud_comp_std = np.std(sp_ompcloud_comp_all, axis=1)
    sp_ompcloud_spark_std = np.std(sp_ompcloud_spark_all, axis=1)
    sp_ompcloud_full_std = np.std(sp_ompcloud_full_all, axis=1)

    sp_ompcloud_comp_mean = np.mean(sp_ompcloud_comp_all, axis=1)
    sp_ompcloud_spark_mean = np.mean(sp_ompcloud_spark_all, axis=1)
    sp_ompcloud_full_mean = np.mean(sp_ompcloud_full_all, axis=1)

    ind = np.arange(N)  # the x locations for the groups
    width = 0.2       # the width of the bars

    fig, ax = plt.subplots()

    errorbar_opt=dict(ecolor='black', lw=1, capsize=2, capthick=1)

    rects_thread = ax.bar(ind, sp_ompthread, width)
    rects_comp = ax.bar(ind + width, sp_ompcloud_comp_mean, width, yerr=sp_ompcloud_comp_std, error_kw=errorbar_opt)
    rects_spark = ax.bar(ind + 2 * width, sp_ompcloud_spark_mean, width, yerr=sp_ompcloud_spark_std, error_kw=errorbar_opt)
    rects_full = ax.bar(ind + 3 * width, sp_ompcloud_full_mean, width, yerr=sp_ompcloud_full_std, error_kw=errorbar_opt)

    # add some text for labels, title and axes ticks
    #ax.set_title(bench)
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Number of cores')
    ax.set_xticks(ind + 1.5 * width)
    ax.set_xticklabels(('8', '16', '32', '64', '128', '192', '256'))

    ax.legend(
        (rects_thread[0], rects_comp[0], rects_spark[0], rects_full[0]),
        ('OpenMP thread', 'OmpCloud-comp', 'OmpCloud-spark', 'OmpCloud-full'),
        loc='upper left')

    plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5)

    pdfFile = os.path.join(path, "speedup_" + bench + ".pdf")

    plt.savefig(pdfFile, format='pdf')


if len(sys.argv) != 2:
    RuntimeException("wrong arguments", false)

print("Generating graphes from " + sys.argv[1] + " .... ")

working_dir = sys.argv[1]

for (suite, benchmarks) in APPLICATION :
    for bench in benchmarks:
        GenGraphes(bench, working_dir)
