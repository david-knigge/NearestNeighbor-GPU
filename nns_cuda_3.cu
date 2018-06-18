#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8
#define THREADS_PER_BLOCK 265 // Number of threads per block

using std::uint32_t; // 32-bit unsigned integer used inside bitvector
using std::size_t;   // unsigned integer for indices

// type for bitvector
typedef array<uint32_t, NW> bitvec_t;
typedef array<size_t, 2> compound_t;
// type for lists of bitvectors
typedef vector<bitvec_t> list_t;
typedef vector<compound_t> output_t;

// type for any function that takes a list_t by reference
typedef void(*callback_list_t)(output_t);

__global__ void nns_kernel(uint32_t *vec_1, uint32_t *vecs, uint32_t *ret_vec, uint32_t *vector_size)
{
    uint32_t wordindex = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t relWordIndex = wordindex % *vector_size;
    uint32_t vectorindex = wordindex / *vector_size;

    if (vectorindex < *vec_1)
    {

        atomicAdd(ret_vec + vectorindex, __popc(vecs[(*vec_1 * *vector_size) + relWordIndex] ^ vecs[wordindex]));

    }
}

__host__ void clearlist(output_t output) {
    for (size_t i = 0; i < output.size(); i++) {
        printf("%zu,", output[i][0]);
        printf("%zu\n", output[i][1]);
    }
}

void NSS(const list_t& L, size_t t, callback_list_t f)  {

    output_t output;
    bitvec_t *vecs;
    uint32_t *vec, *vecd, *vecsd, *ret_vecd, *ret_vec, *vec_size, *vecd_size,
        *ret_vec_zeroes;

    // Initialize Host memory for vectors
    vec = (uint32_t *)malloc(sizeof(uint32_t));
    vecs = (bitvec_t *)malloc(sizeof(bitvec_t) * L.size());
    ret_vec = (uint32_t *)malloc(sizeof(uint32_t) * L.size());
    ret_vec_zeroes = (uint32_t *)calloc(L.size(), sizeof(uint32_t));
    vec_size = (uint32_t *)malloc(sizeof(uint32_t));

    // Copy location of data in vector
    memcpy(vecs, L.data(), L.size() * sizeof(bitvec_t));

    // Set vector size
    *vec_size = L[0].size();

    // Allocate device memory for needed data
    cudaMalloc((void **)&vecd, sizeof(bitvec_t));
    cudaMalloc((void **)&vecsd, L.size() * sizeof(bitvec_t));
    cudaMalloc((void **)&ret_vecd, L.size() * sizeof(uint32_t));
    cudaMalloc((void **)&vecd_size, sizeof(uint32_t));

    // Store L in device memory
    cudaMemcpy(vecsd, vecs, L.size() * sizeof(bitvec_t), cudaMemcpyHostToDevice);

    // Store vector size in device memory
    cudaMemcpy(vecd_size, vec_size, sizeof(uint32_t), cudaMemcpyHostToDevice);

    size_t j;

    for (size_t i = 0; i < L.size(); ++i) {

        // Set current bitvector index
        *vec = i;
        // Initialize device memory to contain id of current bitvector index
        cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Initialize device memory to write found weights to
        cudaMemcpy(ret_vecd, ret_vec_zeroes, L.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

        // Launch kernel
        nns_kernel<<<(((i + 1) * *vec_size) + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(vecd, vecsd, ret_vecd, vecd_size);

        // Retrieve found weights from GPU memory
        cudaMemcpy(ret_vec, ret_vecd, L.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        for (j = 0; j < i; ++j)
        {
            if (ret_vec[j] < t)
            {
                compound_t callback_pair;
                callback_pair[0] = i;
                callback_pair[1] = j;
                output.emplace_back(callback_pair);
            }
        }
        // periodically give outputlist back for further processing
        f(output); // assume it empties output
        output.clear();
    }
    cudaFree(vecd); cudaFree(vecsd); cudaFree(ret_vecd); cudaFree(vecd_size);
    free(vec); free(ret_vec); free(vecs); free(ret_vec_zeroes); free(vec_size);
}

int main() {
    list_t test;
    size_t leng = 10000;
    generate_random_list(test, leng);
    size_t t = 128;

    NSS(test, t, clearlist);

    cout << leng << '\n';
    cout.flush();
    return 0;
}
