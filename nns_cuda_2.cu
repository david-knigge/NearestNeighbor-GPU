#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8
#define THREADS_PER_BLOCK 4 // Number of threads per block

using std::uint32_t; // 32-bit unsigned integer used inside bitvector
using std::size_t;   // unsigned integer for indices

// type for bitvector
typedef array<uint32_t, NW> bitvec_t;
typedef array<size_t, 2> compound_t;
// type for lists of bitvectors
typedef vector<bitvec_t> list_t;
typedef vector<compound_t> output_t;

// type for any function that takes 2 indices
// typedef void(*callback_pair_t)(size_t, size_t);
// type for any function that takes a list_t by reference

typedef void(*callback_list_t)(output_t);

__global__ void cuda_xor(uint32_t *vec_1, uint32_t *vecs, uint32_t *ret_vec)
{
    uint32_t vectorindex = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t wordindex;
    uint32_t vectorweight;

    if (vectorindex < *vec_1)
    {
        vectorweight = 0;
        for (wordindex = 0; wordindex < NW; ++wordindex)
        {
            vectorweight += __popc(vecs[(*vec_1 * NW) + wordindex] ^ vecs[(vectorindex * NW) + wordindex]);
        }
        ret_vec[vectorindex] = vectorweight;
    }
}

inline size_t hammingweight(uint32_t n) {
   return __builtin_popcount(n);
}

void printsomestuff(output_t output) {
    for (size_t i = 0; i < output.size(); i++) {
        for (size_t j = 0; j < output[0].size(); j++) {
            std::bitset<8> x(output[i][j]);
        }
    }
}

void NSS(const list_t& L, size_t t, callback_list_t f)  {

    output_t output;

    // go over all unique pairs 0 <= j < i < L.size()
    bitvec_t *vecs;
    uint32_t *vec, *vecd, *vecsd, *ret_vecd, *ret_vec; // device copies of a, b, c
    //int size = L.size() * sizeof(bitvec_t);
    int size = sizeof(bitvec_t);

    vec = (uint32_t *)malloc(size);
    vecs = (bitvec_t *)malloc(size * L.size());
    ret_vec = (uint32_t *)malloc(L.size() * sizeof(uint32_t));

    memcpy(vecs, L.data(), sizeof(bitvec_t *));

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&vecd, size);
    cudaMalloc((void **)&vecsd, L.size() * size);
    cudaMalloc((void **)&ret_vecd, L.size() * size);
    cudaMemcpy(vecsd, vecs, size * L.size(), cudaMemcpyHostToDevice);

    size_t j;

    for (size_t i = 0; i < L.size(); ++i)    {

        *vec = i;
        cudaMemcpy(vecd, vec, size, cudaMemcpyHostToDevice);

        cuda_xor<<<((i + 1) + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(vecd, vecsd, ret_vecd);
        cudaMemcpy(ret_vec, ret_vecd, size * (i + 1), cudaMemcpyDeviceToHost);

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

    }
    cudaFree(vecd); cudaFree(vecsd); cudaFree(ret_vecd);
    free(vec); free(ret_vec); free(vecs);
}

int main() {
    list_t test;
    size_t leng = 10;
    generate_random_list(test, leng);
    size_t thersh = 98291;
    cout << leng, cout << ' ';
    NSS(test, thersh, printsomestuff);
    cout << "klaar";
    cout.flush();
    return 0;
}
