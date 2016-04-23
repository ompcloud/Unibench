#ifndef KASTORS_MAIN_H
#define KASTORS_MAIN_H

struct user_parameters {
    int check;
    int succeed;
    char* string2display;
    int niter;
    int type;
    int titer;
    int matrix_size;
    int blocksize;
};

extern double run(struct user_parameters* params);

#endif
