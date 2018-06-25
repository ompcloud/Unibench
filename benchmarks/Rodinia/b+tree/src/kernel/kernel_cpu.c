// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#ifdef _OPENMP
#include <omp.h>
#endif             // (in directory known to compiler)			needed by openmp
#include <stdio.h> // (in directory known to compiler)			needed by printf, stderr
#include <stdlib.h> // (in directory known to compiler)			needed by malloc

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h" // (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/timer/timer.h" // (in directory provided here)

//========================================================================================================================================================================================================200
//	KERNEL_CPU FUNCTION
//========================================================================================================================================================================================================200

void kernel_gpu(int cores_arg,

                record *records, knode *knodes, long knodes_elem,
                long records_elem,

                int order, long maxheight, int count,

                long *currKnode, long *offset, int *keys, record *ans) {

  //======================================================================================================================================================150
  //	MCPU SETUP
  //======================================================================================================================================================150

  int max_nthreads;
#ifdef _OPENMP
  max_nthreads = omp_get_max_threads();
  // printf("max # of threads = %d\n", max_nthreads);
  omp_set_num_threads(cores_arg);
// printf("set # of threads = %d\n", cores_arg);
#endif

  int threadsPerBlock;
  threadsPerBlock = order < 1024 ? order : 1024;

  //======================================================================================================================================================150
  //	PROCESS INTERACTIONS
  //======================================================================================================================================================150

  // private thread IDs
  int thid;
  int bid;
  int i;

  int x = 100;
  int *A;
  A = (int *)malloc(sizeof(int) * x);

// process number of querries

#pragma omp target map(                                                        \
    to : keys[ : count],                                                       \
               knodes[ : knodes_elem], records[ : records_elem])               \
                   map(tofrom : offset[ : count],                              \
                                        ans[ : count], currKnode[ : count])
  {
    #pragma omp teams distribute parallel for private(i, thid)
    for (bid = 0; bid < count; bid++) {

      // process levels of the tree
      for (i = 0; i < maxheight; i++) {

        // process all leaves at each level
        for (thid = 0; thid < threadsPerBlock; thid++) {

          // if value is between the two keys
          if ((knodes[currKnode[bid]].keys[thid]) <= keys[bid] &&
              (knodes[currKnode[bid]].keys[thid + 1] > keys[bid])) {
            // this conditional statement is inserted to avoid crush due to but
            // in original code
            // "offset[bid]" calculated below that addresses knodes[] in the
            // next iteration goes outside of its bounds cause segmentation
            // fault
            // more specifically, values saved into knodes->indices in the main
            // function are out of bounds of knodes that they address
            if (knodes[offset[bid]].indices[thid] < knodes_elem) {
              offset[bid] = knodes[offset[bid]].indices[thid];
            }
          }
        }

        // set for next tree level
        currKnode[bid] = offset[bid];
      }

      // At this point, we have a candidate leaf node which may contain
      // the target record.  Check each key to hopefully find the record
      // process all leaves at each level
      for (thid = 0; thid < threadsPerBlock; thid++) {

        if (knodes[currKnode[bid]].keys[thid] == keys[bid]) {
          ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
        }
      }
    }
  }
}

void kernel_cpu(int cores_arg,

                record *records, knode *knodes, long knodes_elem,
                long records_elem,

                int order, long maxheight, int count,

                long *currKnode, long *offset, int *keys, record *ans) {

  //======================================================================================================================================================150
  //	MCPU SETUP
  //======================================================================================================================================================150

  int max_nthreads;
#ifdef _OPENMP
  max_nthreads = omp_get_max_threads();
  // printf("max # of threads = %d\n", max_nthreads);
  omp_set_num_threads(cores_arg);
// printf("set # of threads = %d\n", cores_arg);
#endif

  int threadsPerBlock;
  threadsPerBlock = order < 1024 ? order : 1024;

  //======================================================================================================================================================150
  //	PROCESS INTERACTIONS
  //======================================================================================================================================================150

  // private thread IDs
  int thid;
  int bid;
  int i;

  int x = 100;
  int *A;
  A = (int *)malloc(sizeof(int) * x);

  // process number of querries

  for (bid = 0; bid < count; bid++) {

    // process levels of the tree
    for (i = 0; i < maxheight; i++) {

      // process all leaves at each level
      for (thid = 0; thid < threadsPerBlock; thid++) {

        // if value is between the two keys
        if ((knodes[currKnode[bid]].keys[thid]) <= keys[bid] &&
            (knodes[currKnode[bid]].keys[thid + 1] > keys[bid])) {
          // this conditional statement is inserted to avoid crush due to but in
          // original code
          // "offset[bid]" calculated below that addresses knodes[] in the next
          // iteration goes outside of its bounds cause segmentation fault
          // more specifically, values saved into knodes->indices in the main
          // function are out of bounds of knodes that they address
          if (knodes[offset[bid]].indices[thid] < knodes_elem) {
            offset[bid] = knodes[offset[bid]].indices[thid];
          }
        }
      }

      // set for next tree level
      currKnode[bid] = offset[bid];
    }

    // At this point, we have a candidate leaf node which may contain
    // the target record.  Check each key to hopefully find the record
    // process all leaves at each level
    for (thid = 0; thid < threadsPerBlock; thid++) {

      if (knodes[currKnode[bid]].keys[thid] == keys[bid]) {
        ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
      }
    }
  }
}
