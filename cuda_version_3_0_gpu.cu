#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8
#define THREADS_PER_BLOCK 512 // Number of threads per block
#define NUMBER_OF_THREADS 4096

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
typedef void(*callback_list_t)(output_t *);

__global__ void nns_kernel(uint32_t *start_vec_id, uint32_t *vecs,
                            uint32_t *ret_vec, uint32_t *vector_size,
                            uint32_t *l_size, uint32_t *thres,
                            uint32_t *o_size) {

    // compute which vector the thread has to do the xor operation on
    uint32_t thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t prim_vec = *start_vec_id + thread_id;
    // the variable in which the amount of ones after the xor are added
    uint32_t vectorweight, sec_vec, k, output_index;
    __shared__ unsigned int o_size_block;

    // set the block size variable
    if (threadIdx.x == 0) {
        o_size_block = 0;
        o_size[blockIdx.x] = 0;
    }
    __syncthreads();

    // make sure the vectorindex is within the amount of vectors
    if (prim_vec < *l_size) {
        for (sec_vec = 0; sec_vec < prim_vec; sec_vec++) {
            vectorweight = 0;

            /* for each word in the vector do the xor operation with the
             * corresponding word of the other vector and count the ones
             * with popc
            */
            for (k = 0; k < *vector_size; k++) {
                vectorweight += __popc(vecs[*vector_size * prim_vec + k] ^
                                        vecs[*vector_size * sec_vec + k]);
            }
            // save the vector indices to the right memory place in case they
            // are under the threshold
            if (vectorweight < *thres) {
                output_index = atomicAdd(&o_size_block, 1);
                ret_vec[blockIdx.x * THREADS_PER_BLOCK * *l_size +
                        (output_index * 2)] = prim_vec;
                ret_vec[blockIdx.x * THREADS_PER_BLOCK * *l_size +
                        (output_index * 2) + 1] = sec_vec;
                o_size[blockIdx.x] = output_index + 1;
            }
        }
    }
}

// Takes an output list and prints the indices per line
__host__ void clearlist(output_t *output) {
    for (uint32_t i = 0; i < (*output).size(); i++) {
        total_counter += 1;
        //printf("1: %d  ", output[i][0]);
        //printf("2: %d\n", output[i][1]);
    }
    (*output).clear();
}

void NSS(const list_t& L, uint32_t t, callback_list_t f)  {

    // allocate space for all the variable pointers needed
    output_t output;
    bitvec_t *vecs;
    uint32_t *vec, *vecd, *vecsd, *ret_vecd, *ret_vec, *vec_size, *vecd_size,
        *l_sized, *l_size, *thres, *thresd, *o_size, *o_sized, n_blocks,
        n_threads;

    n_blocks = (NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    n_threads = n_blocks * THREADS_PER_BLOCK;
    l_size = (uint32_t *)malloc(sizeof(uint32_t));
    *l_size = L.size();

    // Initialize Host memory for vectors
    vec = (uint32_t *)malloc(sizeof(uint32_t));
    vecs = (bitvec_t *)malloc(sizeof(bitvec_t) * L.size());
    ret_vec = (uint32_t *)calloc(2 * *l_size *
                (NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1), sizeof(uint32_t));
    vec_size = (uint32_t *)malloc(sizeof(uint32_t));
    thres = (uint32_t *)malloc(sizeof(uint32_t));
    o_size = (uint32_t *)malloc((n_blocks) * sizeof(uint32_t));

    // Copy location of data in vector
    memcpy(vecs, L.data(), *l_size * sizeof(bitvec_t));

    // Set vector size
    *vec_size = L[0].size();
    *thres = t;

    // Allocate device memory for needed data
    cudaMalloc((void **)&vecd, sizeof(bitvec_t));
    cudaMalloc((void **)&vecsd,*l_size * sizeof(bitvec_t));
    cudaMalloc((void **)&ret_vecd, 2 * *l_size *
                (NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1) * sizeof(uint32_t));
    cudaMalloc((void **)&vecd_size, sizeof(uint32_t));
    cudaMalloc((void **)&l_sized, sizeof(uint32_t));
    cudaMalloc((void **)&thresd, sizeof(uint32_t));
    cudaMalloc((void **)&o_sized, (n_blocks) * sizeof(uint32_t));

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
                (vecd, vecsd, ret_vecd, vecd_size, l_sized, thresd, o_sized);

    cudaMemcpy(ret_vec, ret_vecd, 2 * *l_size *
                (NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1) * sizeof(uint32_t),
                cudaMemcpyDeviceToHost);
    cudaMemcpy(o_size, o_sized, n_blocks * sizeof(uint32_t),
                cudaMemcpyDeviceToHost);

    uint32_t j, n_pairs, total_n_pairs, output_back;
    int i;
    for (i = 1 + n_threads; i < *l_size; i = i + n_threads) {
        // Initialize device memory to write found weights to
        *vec = i;

        cudaMemcpy(vecd, vec, sizeof(uint32_t), cudaMemcpyHostToDevice);
        nns_kernel<<< n_blocks, THREADS_PER_BLOCK >>>
                    (vecd, vecsd, ret_vecd, vecd_size,
                     l_sized, thresd, o_sized);

        // compute the amount of indices to be copied back to the cpu
        total_n_pairs = 0;
        for (j = 0; j < n_blocks; j++) {
            total_n_pairs += o_size[j];
        }
        output_back = output.size();
        output.resize(output.size() + total_n_pairs);
        for (j = 0; j < n_blocks; j++) {
            n_pairs = o_size[j];
            memcpy(&output[output_back], ret_vec +
                    (j * THREADS_PER_BLOCK * *l_size),
                    n_pairs * 2 * sizeof(uint32_t));
        }
        // Empty output list
        f(&output);
        output.clear();

        // Retrieve found weights from GPU memory
        cudaMemcpy(ret_vec, ret_vecd, 2 * *l_size *
                    (NUMBER_OF_THREADS + THREADS_PER_BLOCK - 1) *
                    sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(o_size, o_sized, n_blocks * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost);
    }

    // compute the amount of indices to be copied back to the cpu
    total_n_pairs = 0;
    for (j = 0; j < n_blocks; j++) {
        total_n_pairs += o_size[j];
    }
    output_back = output.size();
    output.resize(output.size() + total_n_pairs);
    for (j = 0; j < n_blocks; j++) {
        n_pairs = o_size[j];
        memcpy(&output[output_back], ret_vec +
                (j * THREADS_PER_BLOCK * *l_size),
                n_pairs * 2 * sizeof(uint32_t));
    }
    // Empty output list
    f(&output);
    output.clear();

    cudaFree(vecd); cudaFree(vecsd); cudaFree(ret_vecd); cudaFree(vecd_size);
    cudaFree(l_sized); cudaFree(thresd); cudaFree(o_sized);
    free(vec); free(ret_vec); free(vecs); free(vec_size); free(l_size);
    free(thres); free(o_size);
}

int main() {
    list_t test;
    uint32_t leng = 5000;

    clock_t start;
    double duration;
    start = clock();

    generate_random_list(test, leng);
    uint32_t t = 110;

    NSS(test, t, clearlist);

    duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"printf: "<< duration <<'\n';

    cout << leng << '\n';
    cout << total_counter << '\n';
    cout.flush();
    return 0;
}
