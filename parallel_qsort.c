#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "parallel_qsort.h"

void parallelQsort(int* data, int inputSize, int (*comp)(const void *, const void *)) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(world_size > 0 && (world_size & (world_size - 1)) == 0);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    int loadCount;
    if(world_rank == 0) {
        loadCount = inputSize / world_size;
    }
    MPI_Bcast(&loadCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* local = malloc(sizeof(int) * loadCount);
    MPI_Scatter(data, loadCount, MPI_INT,
                local, loadCount, MPI_INT, 0, MPI_COMM_WORLD);

    for(size_t currentDepth = 0; 1 << currentDepth < world_size; ++currentDepth) {
        size_t numPartners = world_size / (1 << (currentDepth));
        int currentMaster = (world_rank / numPartners) * numPartners;
        int currentPartner = currentMaster +
            (((world_rank - currentMaster) + numPartners/2) % numPartners);
        

        // agree upon pivot
        int numElementsToTheLeft = 0;
        if(currentMaster < world_rank) {
            MPI_Status status;
            MPI_Recv(&numElementsToTheLeft, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD,
                     &status);
        }
        numElementsToTheLeft += loadCount;
        if(world_rank + 1 < currentMaster + numPartners) {
            MPI_Send(&numElementsToTheLeft, 1, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
        }
        
        int pivotIndex;
        if(world_rank == currentMaster + numPartners - 1) {
            if(numElementsToTheLeft > 0) {
                pivotIndex = rand() % numElementsToTheLeft;
            } else {
                pivotIndex = -1;
            }
            for(int i = currentMaster; i < currentMaster + numPartners - 1; ++i) {
                MPI_Send(&pivotIndex, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        } else {
            MPI_Status status;
            MPI_Recv(&pivotIndex, 1, MPI_INT, currentMaster + numPartners - 1, 0, MPI_COMM_WORLD,
                     &status);
        }

        int pivot;
        if(pivotIndex < 0) {
            pivot = 0;
        } else if(numElementsToTheLeft - loadCount <= pivotIndex
                  && pivotIndex < numElementsToTheLeft) {
            pivot = local[pivotIndex - (numElementsToTheLeft - loadCount)];
            
            for(int i = currentMaster; i < currentMaster + numPartners; ++i) {
                if(i != world_rank) {
                    MPI_Send(&pivot, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }
        } else {
             MPI_Status status;
             MPI_Recv(&pivot, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
                      MPI_COMM_WORLD, &status);
        }
        
        // separate into low and high
        int endLow = 0;
        for(int i = 0; i < loadCount; ++i) {
            if(local[i] < pivot) {
                int tmp = local[i];
                local[i] = local[endLow];
                local[endLow] = tmp;
                ++endLow;
            }
        }
        
        if(currentPartner < world_rank) {
            // receive upper
             int oldLoadCount = loadCount;
             int* oldLocal = local;

             MPI_Status status;
             int numReceive;
             MPI_Recv(&numReceive, 1, MPI_INT, currentPartner, MPI_ANY_TAG,
                      MPI_COMM_WORLD, &status);
             
             loadCount = oldLoadCount - endLow + numReceive;
             local = malloc(sizeof(int) * loadCount);
             MPI_Recv(local, numReceive, MPI_INT, currentPartner, MPI_ANY_TAG,
                      MPI_COMM_WORLD, &status);
             memcpy(local + numReceive, oldLocal + endLow, (oldLoadCount - endLow)*sizeof(int));
            
             // send from 0 to (excluding) endLow
             MPI_Send(&endLow, 1, MPI_INT, currentPartner, 0,
                      MPI_COMM_WORLD);
             MPI_Send(oldLocal, endLow, MPI_INT, currentPartner, 0,
                      MPI_COMM_WORLD);
             free(oldLocal);
        } else {
            // send from endLow to (excluding) loadCount
             MPI_Status status;
             int numUpper = loadCount - endLow;
             MPI_Send(&numUpper, 1, MPI_INT, currentPartner, 0,
                      MPI_COMM_WORLD);
             MPI_Send(local + endLow, numUpper, MPI_INT, currentPartner, 0,
                      MPI_COMM_WORLD);

             // receive lower
             int oldLoadCount = loadCount;
             int* oldLocal = local;

             int numReceive;
             MPI_Recv(&numReceive, 1, MPI_INT, currentPartner, MPI_ANY_TAG,
                      MPI_COMM_WORLD, &status);
             
             loadCount = endLow + numReceive;
             local = malloc(sizeof(int) * loadCount);

             memcpy(local, oldLocal, endLow * sizeof(int));
             free(oldLocal);
             MPI_Recv(local + endLow, numReceive, MPI_INT, currentPartner, MPI_ANY_TAG,
                      MPI_COMM_WORLD, &status);
        }
        
#ifdef DEBUG
        if(world_rank == 0) {
            printf("depth = %d:\n", currentDepth);
        }
        for(int i = 0; i < world_size; ++i) {
            if(i == world_rank) {
                printf("%d: numPartners=%d, master=%d/%d partner=%d/%d pivot=%d endLow=%d\n",
                       i, numPartners, currentMaster, world_size,
                       currentPartner, world_size, pivot, endLow);
                printf("\t");
                for(int j = 0; j < loadCount; ++j) {
                    printf("%d ", local[j]);
                }
                printf("\n");
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif // DEBUG
    }
    
    // base case: sort locally
    qsort(local, loadCount, sizeof(int), comp);
    

    // stich results together in root
    int* recvcounts;
    if(world_rank == 0) {
        recvcounts = malloc(sizeof(int) * world_size);
    }
    MPI_Gather(&loadCount, 1, MPI_INT,
               recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int* displs;
    if(world_rank == 0) {
        displs = malloc(sizeof(int) * world_size);
        displs[0] = 0;
        for(int i = 1; i < world_size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    
    MPI_Gatherv(local, loadCount, MPI_INT,
               data, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    free(local);
}
