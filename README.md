UniBench
========

2015 - Institute of Computing, Unicamp, Brazil

__Authors:__ Luís Felipe Mattos, Rafael Cardoso & Márcio Pereira

Introduction
------------

The UniBench is a collection of open source benchmark suites organized in a simple and modular structure. This benchmark suites were converted to make use of the new `target` directive on the OpenMP 4.0, and were developed to test our version of the Clang compiler, which translates OpenMP 4.0 into OpenCL programs.


Quick setup guide
-----------------

To use the UniBench, the script needs the right permissions. If the UniBench script does not have the execution permission, just use:

```
chmod +x ./unibench
```

Use `./parboil help` to display the commands and the available benchmarks.
Use `./parboil list` to display the available benchmarks.


Executing a benchmark
---------------------

The UniBench script offers many options for the execution. If you want to execute a specific benchmark from a specific benchmark suite, you need to specify both the suite and the benchmark in the command line. For example, if you want to compile or execute the 2MM benchmark from the Polybench suite, you just need to use:

```
./unibench compile Polybench/2MM
./unibench run Polybench/2MM
```

The script also accepts the option to execute the complete benchmark suite, you can do this by specifying only the suite in the command line. For example, if you want to compile or execute the Parboil suite, just type:

```
./unibench compile Parboil
./unibench run Parboil
```

However, if you want to compile or execute all the available benchmarks, just use:

```
./unibench compile all
./unibench run all
```

An important point is that this script is __case sensitive__, if you want to be sure that your benchmark name is correct, use the `list` command before executing the UniBench.

For executing this benchmark on mobile phones with Mali GPU devices, just add the flag `-mali`, for example:

```
./unibench compile Parboil -mali
```

For proper compilation when executing on mobile phones, the environment must be working before the compilation!
In this case, binaries, kernels and temporary execution logs will be saved in the directory `/data/local/tmp/<Benchmark Suite>/<Benchmark Name>`.

Cleaning a benchmark
--------------------

The script also offers the `clean` command, with the same options from the previous examples. You can clean the binaries from a specific benchmark from a suite, from a whole suite or from all benchmarks. For example:

```
./unibench clean Polybench/2MM
./unibench clean Parboil
./unibench clean all
```

Output
------

The UniBench outputs the execution time on the GPU device, the execution time for the serial version running on the CPU, and the comparison of the results with an acceptable threshold for the float precision. Here is an example of output for the execution of the Polybench-2MM:

```
<< Linear Algebra: 2 Matrix Multiplications (D=A.B; E=C.D) >>
GPU Runtime: 2.554422s
CPU Runtime: 9.622760s
Non-Matching CPU-GPU Outputs Beyond Error Threshold of 0.05 Percent: 0
```
