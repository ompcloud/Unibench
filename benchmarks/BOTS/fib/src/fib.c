/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include "fib.h"
#include "../../common/BOTSCommonUtils.h"

#define FIB_RESULTS_PRE 41
long long fib_results[FIB_RESULTS_PRE] = {0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393,196418,317811,514229,832040,1346269,2178309,3524578,5702887,9227465,14930352,24157817,39088169,63245986,102334155};

long long int fib_seq (int n)
{
	int x, y;
	if (n < 2) return n;

	x = fib_seq(n - 1);
	y = fib_seq(n - 2);

	return x + y;
}

long long int fib (int n)
{
	long long x, y;
	if (n < 2) return n;

	#pragma omp task untied shared(x) firstprivate(n)
	x = fib(n - 1);
	#pragma omp task untied shared(y) firstprivate(n)
	y = fib(n - 2);

	#pragma omp taskwait
	return x + y;
}

void print_usage() {
   fprintf(stderr, "\n");
   fprintf(stderr, "Usage: %s -[options]\n", "Fibonacci");
   fprintf(stderr, "\n");
   fprintf(stderr, "Where options are:\n");
   fprintf(stderr, "  -n <number>  :  Get Fibonacci number n\n");
   fprintf(stderr, "  -h         : Print program's usage (this help).\n");
}

long long int par_res, seq_res;

int main(int argc, char* argv[]) {
	int n = 20, i;
	for (i=1; i<argc; i++) {
	      if (argv[i][0] == '-') {
	      	switch (argv[i][1]) {
		      	case 'n': /* read argument size 0 */
		               argv[i][1] = '*';
		               i++;
		               if (argc == i) { "Erro\n"; exit(100); }
		               n = atoi(argv[i]);
		               break;
		             case 'h': /* print usage */
		               argv[i][1] = '*';
		               print_usage();
		               exit (100);
		               break;
	           }
	      }
	}

	double t_start, t_end;

	t_start = rtclock();
	#pragma omp parallel
	#pragma omp single
    	par_res = fib(n);
    	t_end = rtclock();
   	fprintf(stdout, "Parallel Runtime: %0.6lfs\n", t_end - t_start);

   	t_start = rtclock();
    	seq_res = fib_seq(n);
    	t_end = rtclock();
   	fprintf(stdout, "Sequential Runtime: %0.6lfs\n", t_end - t_start);

   	if (par_res == seq_res) {
		fprintf(stdout, "Result: Successful\n");
   	} else {
   		fprintf(stdout, "Result: Unsuccessful\n");
   	}

   	return 0;
}
