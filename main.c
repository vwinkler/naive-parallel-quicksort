#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "parallel_qsort.h"

#define DEBUG

int comp(const void* ap, const void* bp) {
    int a = *((int*)ap);
    int b = *((int*)bp);
    
    if(a < b)
        return -1;
    else if(a > b)
        return 1;
    else
        return 0;
}

int main(int argc, char** argv) { 
    MPI_Init(NULL, NULL);
    
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(world_size > 0 && (world_size & (world_size - 1)) == 0);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    // init rng
    srand(time(NULL) + world_rank);
   
    // create input
    int* input;
    int inputSize = 16;
    if(world_rank == 0) {
        input = malloc(sizeof(int) * inputSize);
        for(size_t i = 0; i < inputSize; ++i) {
            input[i] = i;
        }
    }
    
#ifdef DEBUG
    int* inputCopy;
    if(world_rank == 0) {
        inputCopy = malloc(sizeof(int) * inputSize);
        for(size_t i = 0; i < inputSize; ++i) {
            inputCopy[i] = input[i];
        }
    }
#endif // DEBUG
    
    parallelQsort(input, inputSize, comp);

#ifdef DEBUG
    // print results
    if(world_rank == 0) {
        for(size_t i = 0; i < inputSize; ++i) {
            printf(" %d", input[i]);
        }
        
        printf("\n");
        
        qsort(inputCopy, inputSize, sizeof(int), comp);
        int correct = 1;
        for(size_t i = 0; i < inputSize; ++i) {
            correct = correct && input[i] == inputCopy[i];
        }
        printf("correct: %d\n", correct);
        free(inputCopy);
    }
#endif // DEBUG
    
    if(world_rank == 0) {
        free(input);
    }
    
    MPI_Finalize();
}
