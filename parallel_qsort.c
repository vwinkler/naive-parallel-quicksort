#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include "parallel_qsort.h"

typedef struct
{
    int* data;
    int size;
} SortingElement;

void parallelDistributedQsort(SortingElement* local, MPI_Comm communicator,
                              int (*comp)(const void *, const void *)) {
    int numPartners;
    MPI_Comm_size(communicator, &numPartners);
    assert(numPartners > 0 && (numPartners & (numPartners - 1)) == 0);
    
    if(numPartners > 1) {
        int rank;
        MPI_Comm_rank(communicator, &rank);
        
        int currentMaster = 0;
        int currentPartner = currentMaster +
            (((rank - currentMaster) + numPartners/2) % numPartners);

        // agree upon pivot index
        int numElementsToTheLeft = 0;
        MPI_Scan(&local->size, &numElementsToTheLeft, 1, MPI_INT, MPI_SUM, communicator);
        
        int pivotIndex;
        if(rank == currentMaster + numPartners - 1) {
            if(numElementsToTheLeft > 0) {
                pivotIndex = rand() % numElementsToTheLeft;
            } else {
                pivotIndex = -1;
            }
        } 
        MPI_Bcast(&pivotIndex, 1, MPI_INT, 0, communicator);

        // distribute the pivot (value)
        int pivot = INT_MIN;
        if(numElementsToTheLeft - local->size <= pivotIndex
                  && pivotIndex < numElementsToTheLeft) {
            pivot = local->data[pivotIndex - (numElementsToTheLeft - local->size)];
        }
        MPI_Allreduce(MPI_IN_PLACE, &pivot, 1, MPI_INT, MPI_MAX, communicator);
        
        // separate into low and high
        int endLow = 0;
        for(int i = 0; i < local->size; ++i) {
            if(local->data[i] < pivot) {
                int tmp = local->data[i];
                local->data[i] = local->data[endLow];
                local->data[endLow] = tmp;
                ++endLow;
            }
        }
        
        if(currentPartner < rank) {
            // receive upper
             int oldLoadCount = local->size;
             int* oldLocal = local->data;

             MPI_Status status;
             int numReceive;
             MPI_Recv(&numReceive, 1, MPI_INT, currentPartner, MPI_ANY_TAG,
                      communicator, &status);
             
             local->size = oldLoadCount - endLow + numReceive;
             local->data = malloc(sizeof(int) * local->size);
             MPI_Recv(local->data, numReceive, MPI_INT, currentPartner, MPI_ANY_TAG,
                      communicator, &status);
             memcpy(local->data + numReceive, oldLocal + endLow,
                    (oldLoadCount - endLow)*sizeof(int));
            
             // send from 0 to (excluding) endLow
             MPI_Send(&endLow, 1, MPI_INT, currentPartner, 0,
                      communicator);
             MPI_Send(oldLocal, endLow, MPI_INT, currentPartner, 0,
                      communicator);
             free(oldLocal);
        } else {
            // send from endLow to (excluding) local->size
             MPI_Status status;
             int numUpper = local->size - endLow;
             MPI_Send(&numUpper, 1, MPI_INT, currentPartner, 0,
                      communicator);
             MPI_Send(local->data + endLow, numUpper, MPI_INT, currentPartner, 0,
                      communicator);

             // receive lower
             int oldLoadCount = local->size;
             int* oldLocal = local->data;

             int numReceive;
             MPI_Recv(&numReceive, 1, MPI_INT, currentPartner, MPI_ANY_TAG,
                      communicator, &status);
             
             local->size = endLow + numReceive;
             local->data = malloc(sizeof(int) * local->size);

             memcpy(local->data, oldLocal, endLow * sizeof(int));
             free(oldLocal);
             MPI_Recv(local->data + endLow, numReceive, MPI_INT, currentPartner, MPI_ANY_TAG,
                      communicator, &status);
        }
        
#ifdef DEBUG
        for(int i = 0; i < numPartners; ++i) {
            if(i == rank) {
                int world_rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
                printf("%d: rank=%d/%d partner=%d/%d pivot=%d endLow=%d\n",
                       world_rank, rank, numPartners, currentPartner, numPartners, pivot, endLow);
                printf("\t");
                for(int j = 0; j < local->size; ++j) {
                    printf("%d ", local->data[j]);
                }
                printf("\n");
            }
            MPI_Barrier(communicator);
        }
#endif // DEBUG

        MPI_Comm newCommunicator;
        MPI_Comm_split(communicator, (2*rank)/numPartners, 0, &newCommunicator);
        parallelDistributedQsort(local, newCommunicator, comp);
        MPI_Comm_free(&newCommunicator);
    } else {
        // base case: sort locally
        qsort(local->data, local->size, sizeof(int), comp);
    }
}

void parallelQsort(int* input, int inputSize, int (*comp)(const void *, const void *)) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(world_size > 0 && (world_size & (world_size - 1)) == 0);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    assert(inputSize % world_size == 0);
    
    SortingElement local;
    
    if(world_rank == 0) {
        local.size = inputSize / world_size;
    }
    MPI_Bcast(&local.size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local.data = malloc(sizeof(int) * local.size);
    MPI_Scatter(input, local.size, MPI_INT,
                local.data, local.size, MPI_INT, 0, MPI_COMM_WORLD);

    parallelDistributedQsort(&local, MPI_COMM_WORLD, comp);
    

    // stich results together in root
    int* recvcounts;
    if(world_rank == 0) {
        recvcounts = malloc(sizeof(int) * world_size);
    }
    MPI_Gather(&local.size, 1, MPI_INT,
               recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int* displs;
    if(world_rank == 0) {
        displs = malloc(sizeof(int) * world_size);
        displs[0] = 0;
        for(int i = 1; i < world_size; ++i) {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }
    
    MPI_Gatherv(local.data, local.size, MPI_INT,
               input, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    free(local.data);
}
