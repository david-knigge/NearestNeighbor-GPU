#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8
#define THREADS_PER_BLOCK 512 // Number of threads per block

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

__global__ void cuda_xor(uint32_t *vec_1, uint32_t *vec_2, uint32_t *ret_vec)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < NW)
        ret_vec[index] = vec_1[index] + vec_2[index];
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
    for (size_t i = 0; i < L.size(); ++i)    {
        for (size_t j = 0; j < i; ++j)    {
            // compute hamming weight of (L[i] ^ L[j])
            size_t w = 0;

            bitvec_t *vec_1, *vec_2, *ret_vec;
            uint32_t *c_vec_1, *c_vec_2, *c_ret_vec; // device copies of a, b, c
            int size = sizeof(bitvec_t);

            vec_1 = (bitvec_t *)malloc(size);
            vec_2 = (bitvec_t *)malloc(size);
            ret_vec = (bitvec_t *)malloc(size);
            // Allocate space for device copies of a, b, c
            cudaMalloc((void **)&c_vec_1, size);
            cudaMalloc((void **)&c_vec_2, size);
            cudaMalloc((void **)&c_ret_vec, size);

            *vec_1 = L[i];
            *vec_2 = L[j];

            // Copy inputs to device
            cudaMemcpy(c_vec_1, vec_1, size, cudaMemcpyHostToDevice);
            cudaMemcpy(c_vec_2, vec_2, size, cudaMemcpyHostToDevice);
            // Launch add() kernel on GPU with N blocks
            cuda_xor<<<(NW + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(c_vec_1, c_vec_2, c_ret_vec);
            // Copy result back to host
            cudaMemcpy(ret_vec, c_ret_vec, size, cudaMemcpyDeviceToHost);
            // Cleanup
            cudaFree(c_vec_1); cudaFree(c_vec_2); cudaFree(c_ret_vec);

            int l;
            for (l = 0; l < NW; ++l) {
                w += hammingweight((*ret_vec)[l]);
            }

            // if below given threshold then put into output list
            if (w < t)
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
