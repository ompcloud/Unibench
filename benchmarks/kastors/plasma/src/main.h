#ifndef KASTORS_MAIN_H
#define KASTORS_MAIN_H

struct user_parameters {
    int check;
    int succeed;
    char* string2display;
    int niter;
    int type;
    int matrix_size;
    int blocksize;
    int iblocksize;

};

extern double run(struct user_parameters* params);

#endif
