#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8
#define THREADS_PER_BLOCK 256 // Number of threads per block
#define NUMBER_OF_THREADS 1024

using std::uint32_t; // 32-bit unsigned integer used inside bitvector
// using std::size_t;   // unsigned integer for indices

// type for bitvector
typedef array<uint32_t, NW> bitvec_t;
typedef array<uint32_t, 2> compound_t;
// type for lists of bitvectors
typedef vector<bitvec_t> list_t;
typedef vector<compound_t> output_t;

// type for any function that takes a list_t by reference
typedef void(*callback_list_t)(output_t);

// takes in two pointers to the address of two bitvec_t's and a third pointer to
// where the results need to go
__global__ void nns_kernel(uint32_t *start_vec_id, uint32_t *vecs,
                            uint32_t *ret_vec, uint32_t *vector_size,
                            uint32_t *l_size) {

    // compute which vector the thread has to do the xor operation on
    uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t prim_vec = *start_vec_id + thread_id;
    // the variable in which the amount of ones after the xor are added
    uint32_t vectorweight, k;

    // make sure the vectorindex is within the amount of vectors
    if (prim_vec < *l_size)  {
        for (uint32_t j = 0; j < prim_vec; j++) {
            vectorweight = 0;

            /* for each word in the vector do the xor operation with the
             * corresponding word of the other vector and count the ones
             * with popc
            */
            for (wordindex = 0; wordindex < *vector_size; wordindex++) {
                vectorweight += __popc(vecs[*vector_size * prim_vec + wordindex]
                                        ^ vecs[*vector_size * j + wordindex]);
            }
            ret_vec[thread_id * *l_size + j] = vectorweight;
        }
    }
}

// Takes an output list and prints the indices per line
__host__ void print_output(output_t output) {
    for (uint32_t i = 0; i < output.size(); i++) {
        total_counter += 1;
        //printf("%zu,", output[i][0]);
        //printf("%zu\n", output[i][1]);
    }
    output.clear();
}

// takes in a reference to vector full of bitvec_t, an uint32 for the threshold
// and a function for handling the output compares all the vectors in L and does
// Nearest neighbour search.
void NSS(const list_t& L, uint32_t t, callback_list_t f) {

    output_t output;

    // Initialize Host memory for vectors
    bitvec_t *vecs;
    uint32_t *vec, *vecd, *vecsd, *ret_vecd, *ret_vec, *vec_size, *vecd_size,
        *l_sized, *l_size;
    int size = sizeof(bitvec_t);
    l_size = (uint32_t *)malloc(sizeof(uint32_t));
    *l_size = L.size();


    // Initialize Host memory for vectors
    vec = (uint32_t *)malloc(sizeof(uint32_t));
    vecs = (bitvec_t *)malloc(size * *l_size);
    ret_vec = (uint32_t *)malloc(sizeof(uint32_t) * (*l_size * *l_size));
    vec_size = (uint32_t *)malloc(sizeof(uint32_t));
    l_size = (uint32_t *)malloc(sizeof(uint32_t));

    // Copy location of data in vector
    memcpy(vecs, L.data(), *l_size * size);

    // Set vector size
    *vec_size = L[0].size();
    *l_size = L.size();

    // Allocate device memory for needed data
    cudaMalloc((void **)&vecd, size);
    cudaMalloc((void **)&vecsd, *l_size * size);
    cudaMalloc((void **)&vecd_size, sizeof(uint32_t));
    cudaMalloc((void **)&l_sized, sizeof(uint32_t));
    cudaMalloc((void **)&ret_vecd,
                *l_size * *l_size * *l_size * sizeof(uint32_t));

    // Store L in device memory
    cudaMemcpy(vecsd, vecs, *l_size * size, cudaMemcpyHostToDevice);

    // Store vector size in device memory
    cudaMemcpy(vecd_size, vec_size, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Store list size in device memory
    cudaMemcpy(l_sized, l_size, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // start first iteration at vector with index 1
    *vec = 1;
    cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);

    nns_kernel<<< (NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1) /
                    THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
                    (vecd, vecsd, ret_vecd, vecd_size, l_sized);

    // collect the results by copying from the gpu back to the cpu
    cudaMemcpy(ret_vec,
                ret_vecd, *l_size * NUMBER_OF_THREADS * sizeof(uint32_t),
                cudaMemcpyDeviceToHost);

    uint32_t j,prim_vec, sec_vec;
    int i;
    int iterations = *l_size - NUMBER_OF_THREADS;
    for (i = 1 + NUMBER_OF_THREADS; i < iterations; i = i + NUMBER_OF_THREADS) {
        // Initialize device memory to write found weights to
        *vec = i;

        cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);
        nns_kernel<<< (NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1) /
                        THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
                        (vecd, vecsd, ret_vecd, vecd_size, l_sized);

        for (j = 0; j < NUMBER_OF_THREADS; j++) {
            prim_vec = i - NUMBER_OF_THREADS + j;

            if (prim_vec < *l_size) {
                for (sec_vec = 0; sec_vec < prim_vec; sec_vec++) {
                    // check if hit or miss
                    if(ret_vec[(prim_vec - 1) * *l_size + sec_vec] < t) {
                        // create a compound term to add to the output list
                        compound_t callback_pair;
                        callback_pair[0] = prim_vec;
                        callback_pair[1] = sec_vec;
                        output.emplace_back(callback_pair);
                    }
                }
            }
        }

        // Empty output list
        f(output);
        output.clear();

        // Retrieve found weights from GPU memory
        cudaMemcpy(ret_vec, ret_vecd,
                    *l_size * NUMBER_OF_THREADS * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost);

    }

    for (j = 0; j < NUMBER_OF_THREADS; j++) {
        prim_vec = i - NUMBER_OF_THREADS + j;

        if (prim_vec < *l_size) {
            for (sec_vec = 0; sec_vec < prim_vec; sec_vec++) {
                // check if hit or miss
                if(ret_vec[(prim_vec - 1) * *l_size + sec_vec] < t)  {
                    // create a compound term to add to the output list
                    compound_t callback_pair;
                    callback_pair[0] = prim_vec;
                    callback_pair[1] = sec_vec;
                    output.emplace_back(callback_pair);
                }

            }
        }
    }

    // Empty output list
    f(output);
    output.clear();

    cudaFree(vecd); cudaFree(vecsd); cudaFree(ret_vecd); cudaFree(vecd_size);
    free(vec); free(ret_vec); free(vecs); free(vec_size); free(l_size);
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
    cout<<"printf: "<< duration <<'\n';

    cout << leng << '\n';
    cout.flush();
    printf("total pairs:%d\n", total_counter);
    return 0;
}
