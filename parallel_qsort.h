#ifndef PARALLEL_QSORT_H
#define PARALLEL_QSORT_H

void parallelQsort(int* data, int inputSize, int (*comp)(const void *, const void *));

#endif
