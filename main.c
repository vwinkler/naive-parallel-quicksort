#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "parallel_qsort.h"

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
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if(argc != 2) {
        if(world_rank == 0) {
            fprintf(stderr, "illegal number of arguments: %d; expected 1\n", argc);
        }
        MPI_Finalize();
        return 0;
    }
        
    if(world_size <= 0 || (world_size & (world_size - 1)) != 0) {
        if(world_rank == 0) {
            fprintf(stderr, "illegal number of threads: %d; needs to be power of two\n",
                    world_size);
            fprintf(stderr, "aborting\n");
        }
        MPI_Finalize();
        return 0;
    }
    
    // init rng
    if(world_rank == 0) {
        srand(time(NULL) + world_rank);
    }
   
    // create input
    int* input;
    int inputSize = atoi(argv[1]);
    if(world_rank == 0) {
        input = malloc(sizeof(int) * inputSize);
        for(size_t i = 0; i < inputSize; ++i) {
#ifndef DEBUG
            input[i] = rand();
#else
            input[i] = rand() % inputSize;
#endif
        }
    }
    
#ifdef DEBUG
    // print results
    if(world_rank == 0) {
        for(size_t i = 0; i < inputSize; ++i) {
            printf(" %d", input[i]);
        }
        printf("\n");
    }
#endif // DEBUG
    
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
