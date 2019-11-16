#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

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
   
    // create input
    int* input;
    int inputSize = 16;
    if(world_rank == 0) {
        input = malloc(sizeof(int) * inputSize);
        for(size_t i = 0; i < inputSize; ++i) {
            input[i] = i;
        }
    }
    
    int* data;
    int loadCount;
    if(world_rank == 0) {
        assert(inputSize % (2*world_size) == 0);
        loadCount = inputSize / world_size;
        data = malloc(sizeof(int) * inputSize);
        for(size_t i = 0; i < inputSize; ++i) {
            data[i] = input[i];
        }
    }
    

    // start of algorithm ///////////////////////
   

    MPI_Bcast(&loadCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    assert(loadCount % 2 == 0);

    int* local = malloc(sizeof(int) * loadCount);
    MPI_Scatter(data, loadCount, MPI_INT,
                local, loadCount, MPI_INT, 0, MPI_COMM_WORLD);

    for(size_t currentDepth = 0; 1 << currentDepth < world_size; ++currentDepth) {
        size_t numPartners = world_size / (1 << (currentDepth));
        int currentMaster = (world_rank / numPartners) * numPartners;
        int currentPartner = currentMaster +
            (((world_rank - currentMaster) + numPartners/2) % numPartners);
        

        // agree upon pivot
        int pivot;
        if(world_rank == currentMaster) {
            pivot = 0;
            
            for(int i = currentMaster + 1; i < currentMaster + numPartners; ++i) {
                MPI_Send(&pivot, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        } else {
             MPI_Status status;
             MPI_Recv(&pivot, 1, MPI_INT, currentMaster, MPI_ANY_TAG,
                      MPI_COMM_WORLD, &status);
        }
        MPI_Bcast(&pivot, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
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
        
        // begin debug output

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
        
        // end debug output
    }
    
    
    // base case: sort locally
    qsort(local, loadCount, sizeof(int), comp);
    
    // stich results together in root
    MPI_Gather(local, loadCount, MPI_INT,
               data, loadCount * world_size, MPI_INT, 0, MPI_COMM_WORLD);
    

    // end of algorithm /////////////////////////


    // print results
    if(world_rank == 0) {
        for(size_t i = 0; i < loadCount * world_size; ++i) {
            printf(" %d", data[i]);
        }
        
        printf("\n");
        
        qsort(input, inputSize, sizeof(int), comp);
        int correct = 1;
        for(size_t i = 0; i < loadCount * world_size; ++i) {
            correct = correct && input[i] == data[i];
        }
        printf("correct: %d\n", correct);
    }

    free(local);
    
    MPI_Finalize();
}
