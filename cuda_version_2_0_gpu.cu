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

__global__ void nns_kernel(uint32_t *vecs, uint32_t *ret_vec, uint32_t *vector_size, uint32_t *l_size)
{
    uint32_t threadID = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("%d\n", threadID);
    for (uint32_t i = threadID; i < *l_size; i = i+NUMBER_OF_THREADS){
      for (uint32_t j = 0; j<i; j++){
        uint32_t vectorweight = 0;
        // printf("ThreadID: %d prim_vec: %d sec_vec: %d\n",threadID, i, j);
        for (uint32_t k = 0; k < NW; k++){
          vectorweight += __popc(vecs[NW * j + k] ^ vecs[NW * i + k]);
        }
        ret_vec[i * *l_size + j] = vectorweight;
      }
    }
}

__host__ void clearlist(output_t output) {
    for (uint32_t i = 0; i < output.size(); i++) {
        //printf("%zu,", output[i][0]);
        //printf("%zu\n", output[i][1]);
    }
}

void NSS(const list_t& L, uint32_t t, callback_list_t f)  {

    output_t output;
    bitvec_t *vecs;
    uint32_t *vec, *vecd, *vecsd, *ret_vecd, *ret_vec, *vec_size, *vecd_size,
        *l_sized, *l_size;

    // Initialize Host memory for vectors
    vec = (uint32_t *)malloc(sizeof(uint32_t));
    vecs = (bitvec_t *)malloc(sizeof(bitvec_t) * L.size());
    ret_vec = (uint32_t *)malloc(sizeof(uint32_t) * (L.size() * L.size()));
    vec_size = (uint32_t *)malloc(sizeof(uint32_t));
    l_size = (uint32_t *)malloc(sizeof(uint32_t));

    // Copy location of data in vector
    memcpy(vecs, L.data(), L.size() * sizeof(bitvec_t));

    // Set vector size
    *vec_size = L[0].size();
    *l_size = L.size();

    // Allocate device memory for needed data
    cudaMalloc((void **)&vecd, sizeof(bitvec_t));
    cudaMalloc((void **)&vecsd, L.size() * sizeof(bitvec_t));
    cudaMalloc((void **)&ret_vecd, L.size() * L.size() * L.size() * sizeof(uint32_t));
    cudaMalloc((void **)&vecd_size, sizeof(uint32_t));
    cudaMalloc((void **)&l_sized, sizeof(uint32_t));

    // Store L in device memory
    cudaMemcpy(vecsd, vecs, L.size() * sizeof(bitvec_t), cudaMemcpyHostToDevice);

    // Store vector size in device memory
    cudaMemcpy(vecd_size, vec_size, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Store list size in device memory
    cudaMemcpy(l_sized, l_size, sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t i,j;

    // Initialize device memory to write found weights to
    nns_kernel<<< (NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(vecsd, ret_vecd, vecd_size, l_sized);

    // Launch kernel
    // Retrieve found weights from GPU memory
    cudaMemcpy(ret_vec, ret_vecd, L.size() * L.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    for (j = 0; j < L.size(); ++j)
    {
      for (i = 0; i < j; i++)
      {
        if (ret_vec[j * L.size() + i] < t)
        {
            compound_t callback_pair;
            callback_pair[0] = i;
            callback_pair[1] = j;
            output.emplace_back(callback_pair);
        }
      }
    }
    // periodically give outputlist back for further processing
    f(output); // assume it empties output
    output.clear();

    cudaFree(vecd); cudaFree(vecsd); cudaFree(ret_vecd); cudaFree(vecd_size);
    free(vec); free(ret_vec); free(vecs); free(vec_size); free(l_size);
}

int main() {
    list_t test;
    uint32_t leng = 10000;

    clock_t start;
    double duration;
    start = clock();

    generate_random_list(test, leng);
    uint32_t t = 128;

    NSS(test, t, clearlist);

    duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"printf: "<< duration <<'\n';

    cout << leng << '\n';
    cout.flush();
    return 0;
}
