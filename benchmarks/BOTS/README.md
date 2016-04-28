## The Barcelona OpenMP Task Suite (BOTS) Project 

The objective of the suite is to provide a collection of applications that allow to test OpenMP tasking implementations. Most of the kernels come from existing ones from other projects. Each of them comes with different implementations that allow to test different possibilities of the OpenMP task model (task tiedness, cut-offs, single/multiple generators, ...). It currently comes with the following kernels:

Name |	Origin |  Domain | Summary
:---: | :-----:| :-----: | :-----:
Alignment | AKM | Dynamic programming | Aligns sequences of proteins
FFT	| Cilk	| Spectral method	| Computes a Fast Fourier Transformation
Floorplan	| AKM	| Optimization	| Computes the optimal placement of cells in a floorplan
Health	| Olden	| Simulation	| Simulates a country health system
NQueens	| Cilk	| Search	| Finds solutions of the N Queens problem
Sort	| Cilk	| Integer sorting	| Uses a mixture of sorting algorithms to sort a vector
SparseLU		| Sparse linear | algebra	| Computes the LU factorization of a sparse matrix
Strassen	| Cilk	| Dense | linear algebra	| Computes a matrix multiply with Strassen's method

### How to run

Each program was adapted to UniBench. After compiling, you may run a program with the flag -h to see the program's help and see what parameter you can (or must) pass and modify. In some program, you may turn on manual cutoff, if cutoff or final cutoff (this will be said on the help menu). To change the input, just change the INPUT\_FLAGS on the target program's makefile. For example, changing the makefile of the fib program, to "INPUT\_FLAGS=-n 25" would run the program with the 25 as parameter (fibonacci number to be calculated).
