#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

// VERSIE 1.1:
// - zonder threshold op de gpu

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8
#define THREADS_PER_BLOCK 265 // Number of threads per block

using std::uint32_t; // 32-bit unsigned integer used inside bitvector
// using std::size_t;   // unsigned integer for indices

// type for bitvector
typedef array<uint32_t, NW> bitvec_t;
typedef array<uint32_t, 2> compound_t;
// type for lists of bitvectors
typedef vector<bitvec_t> list_t;
typedef vector<compound_t> output_t;


// type for any function that takes a output_t by reference
typedef void(*callback_list_t)(output_t);

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
void print_output(output_t output) {
    for (uint32_t i = 0; i < output.size(); i++) {
        // printf("%d,", output[i][0]);
        // printf("%d\n", output[i][1]);
    }
    output.clear();
}

// takes in a reference to vector full of bitvec_t, an uint32 for the threshold
// and a function for handling the output compares all the vectors in L and does
// Nearest neighbour search.
void NSS(const list_t& L, uint32_t t, callback_list_t f)  {

    output_t output;

    // allocate space for all the variable pointers needed
    bitvec_t *vecs;
    uint32_t *vec, *vecd, *vecsd, *ret_vecd, *ret_vec, *ret_vec_zeroes;
    int size = sizeof(bitvec_t);
    l_size = (uint32_t *)malloc(sizeof(uint32_t));
    *l_size = L.size();


    // allocate space for all the actual values
    vec = (uint32_t *)malloc(sizeof(uint32_t));
    vecs = (bitvec_t *)malloc(size * *l_size);
    ret_vec = (uint32_t *)calloc(*l_size , sizeof(uint32_t));
    ret_vec_zeroes = (uint32_t *)calloc(*l_size, sizeof(uint32_t));

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
    // move the values from the cpu to the gpu
    cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ret_vecd, ret_vec_zeroes, L.size() * sizeof(uint32_t),
                cudaMemcpyHostToDevice);
    // run 1 kernel
    cuda_xor<<<1, THREADS_PER_BLOCK>>>(vecd, vecsd, ret_vecd);
    // collect the results by copying from the gpu back to the cpu
    cudaMemcpy(ret_vec, ret_vecd, L.size() * sizeof(uint32_t),
                cudaMemcpyDeviceToHost);

    for (i = 1; i < *l_size; ++i) {

        *vec = i;
        cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(ret_vecd, ret_vec_zeroes, *l_size * sizeof(uint32_t),
                    cudaMemcpyHostToDevice);

        // apply the cuda_xor function to all the up to i vectors
        cuda_xor<<<(i + THREADS_PER_BLOCK) / THREADS_PER_BLOCK,
                    THREADS_PER_BLOCK>>>(vecd, vecsd, ret_vecd);

        for (j = 0; j < i-1; j++) {
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
        f(output);
        output.clear(); // clear the output

    }
    // free the allocated memmory
    cudaFree(vecd); cudaFree(vecsd); cudaFree(ret_vecd);
    free(vec); free(ret_vec); free(vecs); free(ret_vec_zeroes);
}

int main() {
    list_t test;
    uint32_t leng = 10000;

    clock_t start;
    double duration;
    start = clock();

    generate_random_list(test, leng);
    uint32_t thersh = 110;
    cout << leng, cout << '\n';

    NSS(test, thersh, print_output);

    duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"printf: "<< duration <<'\n';

    cout << "klaar\n";
    cout.flush();
    return 0;
}
