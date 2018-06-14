#include <random>
#include <iostream>
#include <array>
#include <vector>
#include <cstdint>

using namespace std;

#define NW 2048 // use bitvectors of d=NW*32 bits, example NW=8

using std::uint32_t; // 32-bit unsigned integer used inside bitvector
using std::size_t;   // unsigned integer for indices

// type for bitvector
typedef array<uint32_t, NW> bitvec_t;
typedef array<size_t, 2> compound_t;
// type for lists of bitvectors
typedef vector<bitvec_t> list_t;
typedef vector<compound_t> output_t;
// type for any function that takes 2 indices
typedef void(*callback_pair_t)(size_t, size_t);
// type for any function that takes a list_t by reference
typedef void(*callback_list_t)(output_t);




void generate_random_list(list_t& output, size_t n) {
    // a true randomness source
    random_device rd;
    // obtain randomness seed, replace by a fixed value for deterministic tests
    uint32_t seed = rd();
    // a fast pseudo-random generator
    // each mt() call returns a pseudo-random uint32_t value
    mt19937 mt(seed);
    // resize output to hold n elements
    output.resize(n);
    // set random value for each element
<<<<<<< HEAD
    for (size_t i = 0; i < n; i++)  {
=======
    for (size_t i = 0; i < n; ++i)  {
>>>>>>> e975a79ffb9bbc6d17a9b6a1f8ee35fbf2380371
        for (size_t k = 0; k < NW; ++k)
            output[i][k] = mt();
    }
    // output list is given by reference, so nothing to return
}
