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

#plt.rcParams.update({'font.size': 12})

#Nbcore = [8, 16, 32, 64, 128, 192, 256]
nbcore2ind = dict([("1", -1), ("8", 0), ("16", 1), ("32", 2), ("64", 3), ("128", 4), ("192", 5), ("256", 6)])

all_overhead_openmp = np.zeros(8*2)
all_overhead_openmp_spark = np.zeros(8*2)
all_overhead_openmp_total = np.zeros(8*2)

def GenGraphes(bench, path, bench_index):

    print(bench)

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

    #print(sp_ompcloud_comp_mean)
    #print(sp_ompcloud_spark_mean)
    #print(sp_ompcloud_full_mean)

    sp_ompcloud_comp_list = np.hsplit(sp_ompcloud_comp_all, [1])
    sp_ompcloud_spark_list = np.hsplit(sp_ompcloud_spark_all, [1])
    sp_ompcloud_full_list = np.hsplit(sp_ompcloud_full_all, [1])

    ind = np.arange(N)  # the x locations for the groups

    width = 0.3       # the width of the bars

    ## Generate speedup graphs
    fig, ax = plt.subplots()

    rects_comp = ax.bar(ind, sp_ompcloud_comp_list[0], width)
    rects_spark = ax.bar(ind + width, sp_ompcloud_spark_list[0], width)
    rects_full = ax.bar(ind + 2 * width, sp_ompcloud_full_list[0], width)

    # add some text for labels, title and axes ticks
    #ax.set_title(bench)
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Number of cores')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('8', '16', '32', '64', '128', '192', '256'))

    plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5)

    pdfFile = os.path.join(path, "speedup_sparse_" + bench + ".pdf")

    plt.savefig(pdfFile, format='pdf')

    ## Generate speedup graphs
    fig, ax = plt.subplots()

    rects_comp = ax.bar(ind + width, sp_ompcloud_comp_list[1], width)
    rects_spark = ax.bar(ind + 2 * width, sp_ompcloud_spark_list[1], width)
    rects_full = ax.bar(ind + 3 * width, sp_ompcloud_full_list[1], width)

    # add some text for labels, title and axes ticks
    #ax.set_title(bench)
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Number of cores')
    ax.set_xticks(ind + 1.5 * width)
    ax.set_xticklabels(('8', '16', '32', '64', '128', '192', '256'))

    plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5)

    pdfFile = os.path.join(path, "speedup_dense_" + bench + ".pdf")

    plt.savefig(pdfFile, format='pdf')

    ## Generate speedup graphs
    fig, ax = plt.subplots()

    width = 0.2       # the width of the bars

    errorbar_opt=dict(ecolor='black', lw=1, capsize=2, capthick=1)

    rects_thread = ax.bar(ind, sp_ompthread, width, color='C3')
    rects_comp = ax.bar(ind + width, sp_ompcloud_comp_mean, width)
    rects_spark = ax.bar(ind + 2 * width, sp_ompcloud_spark_mean, width)
    rects_full = ax.bar(ind + 3 * width, sp_ompcloud_full_mean, width)

    # add some text for labels, title and axes ticks
    #ax.set_title(bench)
    ax.set_ylabel('Speedup')
    ax.set_xlabel('Number of cores')
    ax.set_xticks(ind + 1.5 * width)
    ax.set_xticklabels(('8', '16', '32', '64', '128', '192', '256'))

    #ax.legend(
    #    (rects_thread[0], rects_comp[0], rects_spark[0], rects_full[0], rects_comp.errorbar),
    #    ('OmpThread', 'OmpCloud-computation', 'OmpCloud-spark', 'OmpCloud-full', 'Variation sparse/dense'),
    #    loc='upper left')

    plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5)

    pdfFile = os.path.join(path, "speedup_" + bench + ".pdf")

    plt.savefig(pdfFile, format='pdf')


    ## Generate speedup graphs
    fig2, ax2 = plt.subplots()

    width = 0.4

    # Compute the overhead
    ompcloud_sparkoverhead_all = np.subtract(ompcloud_spark_all,ompcloud_comp_all)
    ompcloud_commoverhead_all = np.subtract(ompcloud_full_all,ompcloud_spark_all)

    ompcloud_comp_mean = np.mean(ompcloud_comp_all, axis=1)
    ompcloud_spark_mean = np.mean(ompcloud_spark_all, axis=1)
    ompcloud_full_mean = np.mean(ompcloud_full_all, axis=1)

    ompcloud_full_list = np.hsplit(ompcloud_full_all, [1])
    print(np.divide(ompcloud_full_list,60))

    overhead_openmp = np.divide(np.subtract(ompthread, ompcloud_comp_mean), ompthread)
    overhead_openmp_spark = np.divide(np.subtract(ompthread, ompcloud_spark_mean), ompthread)
    overhead_openmp_total = np.divide(np.subtract(ompthread, ompcloud_full_mean), ompthread)

    #all_overhead_openmp.append(overhead_openmp[0])
    #print(overhead_openmp)


    all_overhead_openmp[2*bench_index] = overhead_openmp[0]
    all_overhead_openmp[2*bench_index+1] = overhead_openmp[1]
    all_overhead_openmp_spark[2*bench_index] = overhead_openmp_spark[0]
    all_overhead_openmp_spark[2*bench_index+1] = overhead_openmp_spark[1]
    all_overhead_openmp_total[2*bench_index] = overhead_openmp_total[0]
    all_overhead_openmp_total[2*bench_index+1] = overhead_openmp_total[1]
    #all_overhead_openmp.append(overhead_openmp[1])

    #print(np.divide(np.subtract(ompcloud_spark_mean,ompcloud_comp_mean),ompcloud_spark_mean))
    #print(np.divide(np.subtract(ompcloud_full_mean,ompcloud_spark_mean),ompcloud_spark_mean))

    ompcloud_comp_list = np.hsplit(ompcloud_comp_all, [1])
    #print (ompcloud_comp_all)
    #print (ompcloud_comp_list)

    ompcloud_sparkoverhead_list = np.hsplit(ompcloud_sparkoverhead_all, [1])
    ompcloud_commoverhead_list = np.hsplit(ompcloud_commoverhead_all, [1])

    rects_comp_sparse = ax2.bar(ind, ompcloud_comp_list[0], width)
    rects_sparkoverhead_sparse = ax2.bar(ind, ompcloud_sparkoverhead_list[0], width, bottom=ompcloud_comp_list[0])
    rects_commoverhead_sparse = ax2.bar(ind, ompcloud_commoverhead_list[0], width, bottom=ompcloud_sparkoverhead_list[0]+ompcloud_comp_list[0])

    rects_comp_full = ax2.bar(ind + 1.1 * width, ompcloud_comp_list[1], width, color='#185088')
    rects_sparkoverhead_full = ax2.bar(ind + 1.1 * width, ompcloud_sparkoverhead_list[1], width, bottom=ompcloud_comp_list[1], color='#CB4813')
    rects_commoverhead_full = ax2.bar(ind + 1.1 * width, ompcloud_commoverhead_list[1], width, bottom=ompcloud_sparkoverhead_list[1]+ompcloud_comp_list[1], color='#336D2E')

    # add some text for labels, title and axes ticks
    #ax.set_title(bench)
    ax2.set_ylabel('Time in second')
    ax2.set_xlabel('Number of cores')
    ax2.set_xticks(ind + 0.55 * width)
    ax2.set_xticklabels(('8', '16', '32', '64', '128', '192', '256'))

    categories = ['sparse', 'dense']

    #leg3 = legend([p5, p1, p2, p5, p3, p4],
    #          [r'$D_{etc}$'] + categories + [r'$A_{etc}$'] + categories,
    #          loc=2, ncol=2) # Two columns, vertical group labels

    #ax2.legend(
    #    [rects_comp_sparse[0], rects_comp_full[0],
    #    rects_sparkoverhead_sparse[0], rects_sparkoverhead_full[0],
    #    rects_commoverhead_sparse[0], rects_commoverhead_full[0]],
    #    ['Computation time (sparse)','Computation time (dense)',
    #    'Spark overhead (sparse)','Spark overhead (dense)',
    #    'Host-Target communication (sparse)','Host-Target communication (dense)'],
    #    loc='upper right') # Two columns, vertical group labels

    plt.grid(axis='y', color='grey', linestyle='-', linewidth=0.5)

    pdfFile = os.path.join(path, "exec_" + bench + ".pdf")

    plt.savefig(pdfFile, format='pdf')


if len(sys.argv) != 2:
    RuntimeException("wrong arguments", false)

print("Generating graphes from " + sys.argv[1] + " .... ")

working_dir = sys.argv[1]

bench_index=0
for (suite, benchmarks) in APPLICATION :
    for bench in benchmarks:
        GenGraphes(bench, working_dir,bench_index)
        bench_index = bench_index+1

print(all_overhead_openmp)

print(np.median(all_overhead_openmp))
print(np.median(all_overhead_openmp_spark))
print(np.median(all_overhead_openmp_total))
