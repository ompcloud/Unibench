#ifndef KASTORS_MAIN_H
#define KASTORS_MAIN_H

struct user_parameters {
    int check;
    int succeed;
    char* string2display;
    int niter;
    int type;
#ifdef TITER
    int titer;
#endif

    int matrix_size;
    int submatrix_size;

#ifdef BSIZE
    int blocksize;
#endif
#ifdef IBSIZE
    int iblocksize;
#endif
#ifdef CUTOFF_DEPTH
    int cutoff_depth;
#endif
#ifdef CUTOFF_SIZE
    int cutoff_size;
#endif
};

extern double run(struct user_parameters* params);

#endif
