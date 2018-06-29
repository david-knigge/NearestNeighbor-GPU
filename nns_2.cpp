#include <array>
#include <vector>
#include <bitset>

#include <cstdint>
#include <iostream>
#include "./generate_data.cpp"

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8

int comparisons = 0, under_thresh = 0;
// comparisons = 0;
// under_thresh = 0;

using namespace std;

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

inline size_t hammingweight(uint32_t n) {
   return __builtin_popcount(n);
}

// 1: hoeveelheid overeenkomsten
// 2: eerste bitstring
// 3: tweede bitstring
void printsomestuff(output_t output) {
    for (size_t i = 0; i < output.size(); i++) {
        for (size_t j = 0; j < output[0].size(); j++) {
            // std::bitset<8> x(output[i][j]);
            // cout << x, cout << ' ';
        }
        under_thresh++;
         // cout << '\n';
    }
}

void NSS(const list_t& L, size_t t, callback_list_t f)  {

    output_t output;

    // go over all unique pairs 0 <= j < i < L.size()
    for (size_t i = 0; i < L.size(); ++i)    {
        for (size_t j = 0; j < i; ++j)    {
            comparisons++;
            // compute hamming weight of (L[i] ^ L[j])
            size_t w = 0;
            for (size_t k = 0; k < NW; ++k) {
              w += hammingweight(L[i][k] ^ L[j][k]);
              // std::bitset<8> x(w);

              // cout << w, cout << ' ',cout << L[i][k], cout << ' ', cout << L[j][k], cout << '\n';
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
        output.clear();

    }
}

int main() {
    list_t test;
    size_t leng = 50000;
    generate_random_list(test, leng);
    size_t thersh = 110;
    cout << leng, cout << ' ';
    NSS(test, thersh, printsomestuff);
    cout << "klaar\n";
    cout << comparisons;
    cout << '\n';
    cout << under_thresh;
    cout.flush();
    return 0;
}
