#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

// VERSIE 1.2:
// - met threshold op de gpu

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

// type for any function that takes 2 indices
// typedef void(*callback_pair_t)(size_t, size_t);
// type for any function that takes a list_t by reference

typedef void(*callback_list_t)(output_t);

__global__ void nns_kernel(uint32_t *vec_1, uint32_t *vecs, uint32_t *ret_vec, uint32_t *size, uint32_t *thres)
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
        if (vectorweight < *thres) {
          ret_vec[vectorindex] = *vec_1;
          ret_vec[vectorindex+ *size] = vectorindex;
        }
    }
}

inline size_t hammingweight(uint32_t n) {
   return __builtin_popcount(n);
}

void printsomestuff(output_t output) {
    for (size_t i = 0; i < output.size(); i++) {
        printf("%zu,", output[i][0]);
        printf("%zu\n", output[i+1][1]);
    }
    output.clear();
}

void NSS(const list_t& L, uint32_t *t, callback_list_t f)  {

    output_t output;

    // go over all unique pairs 0 <= j < i < L.size()
    bitvec_t *vecs;
    uint32_t *vec, *vecd, *vecsd, *ret_vecd, *ret_vec, *ret_vec_zeroes, *l_sized, *l_size, *thresd;
    int size = sizeof(bitvec_t);

    vec = (uint32_t *)malloc(sizeof(uint32_t));
    vecs = (bitvec_t *)malloc(sizeof(bitvec_t) * L.size());
    ret_vec = (uint32_t *)calloc(2 * L.size() , sizeof(uint32_t));
    ret_vec_zeroes = (uint32_t *)calloc(L.size(), sizeof(uint32_t));
    l_size = (uint32_t *)malloc(sizeof(uint32_t));

    memcpy(vecs, L.data(), L.size() * sizeof(bitvec_t));

    *l_size = L.size();

    // Allocate space for device copies of our primary vector, our entire setup
    // of vectors
    cudaMalloc((void **)&vecd, size);
    cudaMalloc((void **)&vecsd, L.size() * size);
    cudaMalloc((void **)&ret_vecd, 2 * L.size() * sizeof(uint32_t));
    cudaMemcpy(vecsd, vecs, L.size() * size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&thresd, sizeof(uint32_t));
    cudaMalloc((void **)&l_sized, sizeof(uint32_t));

    cudaMemcpy(thresd, t, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(l_sized, l_size, sizeof(uint32_t), cudaMemcpyHostToDevice);


    // run 1 kernel
    nns_kernel<<<1, THREADS_PER_BLOCK>>>(vecd, vecsd, ret_vecd, l_sized, thresd);
    cudaMemcpy(ret_vec, ret_vecd, L.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (size_t i = 1; i < L.size(); ++i)    {

        *vec = i;
        cudaMemcpy(ret_vecd, ret_vec_zeroes, L.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);

        nns_kernel<<<(i + THREADS_PER_BLOCK) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(vecd, vecsd, ret_vecd, l_sized, thresd);

        compound_t callback_pair;
        callback_pair[0] = ret_vec[i];
        callback_pair[1] = ret_vec[i+L.size()];
        output.emplace_back(callback_pair);

        cudaMemcpy(ret_vec, ret_vecd, L.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // periodically give outputlist back for further processing
        f(output); // assume it empties output
        output.clear();

    }
    cudaFree(vecd); cudaFree(vecsd); cudaFree(ret_vecd);
    free(vec); free(ret_vec); free(vecs); free(ret_vec_zeroes);
}

int main() {
    list_t test;
    size_t leng = 5000;

    clock_t start;
    double duration;
    start = clock();

    generate_random_list(test, leng);

    // threshold
    uint32_t *t;
    t = (uint32_t *)malloc(sizeof(uint32_t));
    *t = 128;

    cout << leng, cout << '\n';

    NSS(test, t, printsomestuff);

    duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"printf: "<< duration <<'\n';

    cout << "klaar\n";
    cout.flush();
    return 0;
}