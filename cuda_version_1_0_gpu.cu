#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

// VERSIE 1.1:
// - zonder threshold op de gpu

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8
#define THREADS_PER_BLOCK 512 // Number of threads per block

int total_counter = 0;
using std::uint32_t; // 32-bit unsigned integer used inside bitvector
// using std::size_t;   // unsigned integer for indices

// type for bitvector
typedef array<uint32_t, NW> bitvec_t;
typedef array<uint32_t, 2> compound_t;
// type for lists of bitvectors
typedef vector<bitvec_t> list_t;
typedef vector<compound_t> output_t;


// type for any function that takes a output_t by reference
typedef void(*callback_list_t)(output_t *);

// takes in two pointers to the address of two bitvec_t's and a third pointer to
// where the results need to go
__global__ void cuda_xor(uint32_t *vec_1, uint32_t *vecs, uint32_t *ret_vec) {

    // compute which vector the thread has to do the xor operation on
    uint32_t vectorindex = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t wordindex;
    // the variable in which the amount of ones after the xor are added
    uint32_t vectorweight;

    // make sure the vectorindex is within the amount of vectors
    if (vectorindex < *vec_1) {
        vectorweight = 0;

        /* for each word in the vector do the xor operation with the
         * corresponding word of the other vector and count the ones
         * with popc
        */
        for (wordindex = 0; wordindex < NW; ++wordindex) {
            vectorweight += __popc(vecs[(*vec_1 * NW) + wordindex] ^
                                    vecs[(vectorindex * NW) + wordindex]);
        }
        ret_vec[vectorindex] = vectorweight;
    }
}

// Takes an output list and prints the indices per line
void print_output(output_t *output) {
    for (uint32_t i = 0; i < (*output).size(); i++) {
        total_counter += 1;
        //printf("1: %d  ", output[i][0]);
        //printf("2: %d\n", output[i][1]);
    }
    (*output).clear();
}

// takes in a reference to vector full of bitvec_t, an uint32 for the threshold
// and a function for handling the output compares all the vectors in L and does
// Nearest neighbour search.
void NSS(const list_t& L, uint32_t t, callback_list_t f)  {

    output_t output;

    // allocate space for all the variable pointers needed
    bitvec_t *vecs;
    uint32_t *vec, *vecd, *vecsd, *ret_vecd, *ret_vec, *l_size;
    //int size = L.size() * sizeof(bitvec_t);
    int size = sizeof(bitvec_t);
    l_size = (uint32_t *)malloc(sizeof(uint32_t));
    *l_size = L.size();


    // allocate space for all the actual values
    vec = (uint32_t *)malloc(sizeof(uint32_t));
    vecs = (bitvec_t *)malloc(sizeof(bitvec_t) * L.size());
    ret_vec = (uint32_t *)calloc(L.size(), sizeof(uint32_t));

    memcpy(vecs, L.data(), *l_size * size);

    // Allocate space for device copies of our primary vector, our entire setup
    // of vectors
    cudaMalloc((void **)&vecd, size);
    cudaMalloc((void **)&vecsd, *l_size * size);
    cudaMalloc((void **)&ret_vecd, *l_size * sizeof(uint32_t));
    cudaMemcpy(vecsd, vecs, *l_size * size, cudaMemcpyHostToDevice);

    // allocate space fo vector indices
    uint32_t i,j;

    *vec = 1;
    cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // run 1 kernel
    cuda_xor<<<1, THREADS_PER_BLOCK>>>(vecd, vecsd, ret_vecd);
    // collect the results by copying from the gpu back to the cpu
    cudaMemcpy(ret_vec, ret_vecd, L.size() * sizeof(uint32_t),
                cudaMemcpyDeviceToHost);

    for (i = 1; i < *l_size; ++i) {

        *vec = i;
        cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);

        // apply the cuda_xor function to all the up to i vectors
        cuda_xor<<<(i + THREADS_PER_BLOCK) / THREADS_PER_BLOCK,
                    THREADS_PER_BLOCK>>>(vecd, vecsd, ret_vecd);

        for (j = 0; j < i - 1; j++) {
            if (ret_vec[j] < t) {
                // create a compound term to add to the output list
                compound_t callback_pair;
                callback_pair[0] = i;
                callback_pair[1] = j;
                output.emplace_back(callback_pair);
            }
        }
        // collect the results by copying from the gpu back to the cpu
        cudaMemcpy(ret_vec, ret_vecd, *l_size * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost);

        // periodically give outputlist back for further processing
        f(&output);

    }

    for (j = 0; j < i - 1; j++) {
        if (ret_vec[j] < t) {
            // create a compound term to add to the output list
            compound_t callback_pair;
            callback_pair[0] = i;
            callback_pair[1] = j;
            output.emplace_back(callback_pair);
        }
    }

    // Empty output list
    f(&output);
    // free the allocated memmory
    cudaFree(vecd); cudaFree(vecsd); cudaFree(ret_vecd);
    free(vec); free(ret_vec); free(vecs);
}

int main() {
    list_t test;
    uint32_t leng = 5000;

    // starting the timer
    clock_t start;
    double duration;
    start = clock();

    // generating the dataset
    generate_random_list(test, leng);
    // setting the threshold
    uint32_t t = 110;

    NSS(test, t, print_output);

    // end the timer
    duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"execution duration: "<< duration <<'\n';
    cout<<"total pairs: " << total_counter << '\n';
    cout.flush();
    return 0;
}
