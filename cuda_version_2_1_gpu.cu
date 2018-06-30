#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8
#define THREADS_PER_BLOCK 256 // Number of threads per block
#define NUMBER_OF_THREADS 2048

using std::uint32_t; // 32-bit unsigned integer used inside bitvector
// using std::size_t;   // unsigned integer for indices

int total_counter = 0;

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
                            uint32_t *l_size, uint32_t *thres) {

    // compute which vector the thread has to do the xor operation on
    uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t prim_vec = *start_vec_id + thread_id;
    // the variable in which the amount of ones after the xor are added
    uint32_t vectorweight, k;

    // make sure the vectorindex is within the amount of vectors
    if (prim_vec < *l_size) {
        for (uint32_t j = 0; j < prim_vec; j++) {
            vectorweight = 0;

            /* for each word in the vector do the xor operation with the
             * corresponding word of the other vector and count the ones
             * with popc
            */
            for (k = 0; k < *vector_size; k++) {
                vectorweight += __popc(vecs[*vector_size * prim_vec + k] ^
                                        vecs[*vector_size * j + k]);
            }
            // thresholding the weight on the gpu
            ret_vec[thread_id * *l_size + j] = (vectorweight < *thres);
        }
    }
}

// Takes an output list and prints the indices per line
__host__ void print_output(output_t output) {
    for (uint32_t i = 0; i < output.size(); i++) {
        //printf("%d,", output[i][0]);
        //printf("%d\n", output[i][1]);
        total_counter += 1;
    }
    output.clear();
}

// takes in a reference to vector full of bitvec_t, an uint32 for the threshold
// and a function for handling the output compares all the vectors in L and does
//  Nearest neighbour search.
void NSS(const list_t& L, uint32_t t, callback_list_t f)  {

    // allocate space for all the variable pointers needed
    output_t output;
    bitvec_t *vecs;
    uint32_t *vec, *vecd, *vecsd, *ret_vecd, *ret_vec, *vec_size, *vecd_size,
        *l_sized, *l_size, *thres, *thresd, n_blocks, n_threads;

    n_blocks = (NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    n_threads = n_blocks * THREADS_PER_BLOCK;
    l_size = (uint32_t *)malloc(sizeof(uint32_t));
    *l_size = L.size();

    // Initialize Host memory for vectors
    vec = (uint32_t *)malloc(sizeof(uint32_t));
    vecs = (bitvec_t *)malloc(sizeof(bitvec_t) * *l_size);
    ret_vec = (uint32_t *)calloc(*l_size *n_threads, sizeof(uint32_t));
    vec_size = (uint32_t *)malloc(sizeof(uint32_t));
    thres = (uint32_t *)malloc(sizeof(uint32_t));

    // Copy location of data in vector
    memcpy(vecs, L.data(), *l_size * sizeof(bitvec_t));

    // Set vector size
    *vec_size = L[0].size();
    *thres = t;

    // Allocate device memory for needed data
    cudaMalloc((void **)&vecd, sizeof(bitvec_t));
    cudaMalloc((void **)&vecsd,*l_size * sizeof(bitvec_t));
    cudaMalloc((void **)&vecd_size, sizeof(uint32_t));
    cudaMalloc((void **)&l_sized, sizeof(uint32_t));
    cudaMalloc((void **)&thresd, sizeof(uint32_t));
    cudaMalloc((void **)&ret_vecd,
                *l_size * n_threads * sizeof(uint32_t));

    // Store L in device memory
    cudaMemcpy(vecsd, vecs, *l_size * sizeof(bitvec_t), cudaMemcpyHostToDevice);

    // Store vector size in device memory
    cudaMemcpy(vecd_size, vec_size, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Store list size in device memory
    cudaMemcpy(l_sized, l_size, sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMemcpy(thresd, thres, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // start first iteration at vector with index 1
    *vec = 1;
    cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);

    nns_kernel<<< n_blocks, THREADS_PER_BLOCK>>>
                    (vecd, vecsd, ret_vecd, vecd_size, l_sized, thresd);

    cudaMemcpy(ret_vec, ret_vecd,
                *l_size * n_threads * sizeof(uint32_t),
                cudaMemcpyDeviceToHost);

    uint32_t j,prim_vec, sec_vec;
    int i;
    for (i = 1 + n_threads; i < *l_size; i = i + n_threads) {
        // Initialize device memory to write found weights to
        *vec = i;

        cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);
        nns_kernel<<< n_blocks, THREADS_PER_BLOCK >>>
                        (vecd, vecsd, ret_vecd, vecd_size, l_sized, thresd);

        for (j = 0; j < n_threads; j++) {
            prim_vec = i - n_threads + j;

            if (prim_vec < *l_size) {
                for (sec_vec = 0; sec_vec < prim_vec; sec_vec++) {
                    // check if hit or miss
                    if(ret_vec[j * *l_size + sec_vec]) {
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
                    *l_size * n_threads * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost);

    }

    for (j = 0; j < n_threads; j++) {
        prim_vec = i - n_threads + j;

        if (prim_vec < *l_size) {
            for (sec_vec = 0; sec_vec < prim_vec; sec_vec++) {
                // check if hit or miss
                if(ret_vec[j * *l_size + sec_vec]) {
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
    cudaFree(l_sized); cudaFree(thresd);
    free(vec); free(ret_vec); free(vecs); free(vec_size); free(l_size);
    free(thres);
}

int main() {
    list_t test;
    uint32_t leng = 10000;

    clock_t start;
    double duration;
    start = clock();

    generate_random_list(test, leng);
    uint32_t t = 110;

    NSS(test, t, print_output);

    duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"printf: "<< duration <<'\n';

    cout << leng << '\n';
    printf("total pairs:%d\n", total_counter);
    cout.flush();
    return 0;
}
