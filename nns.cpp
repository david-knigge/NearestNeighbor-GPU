#include <array>
#include <vector>
#include <cstdint>
#include <iostream>
#include <deque>

#define NW 8 // use bitvectors of d=NW*32 bits, example NW=8

using namespace std;

using std::uint32_t; // 32-bit unsigned integer used inside bitvector
using std::size_t;   // unsigned integer for indices

// type for bitvector
typedef array<std::uint32_t, NW> bitvec_t;
// type for lists of bitvectors
typedef vector<bitvec_t> list_t;
// type for any function that takes 2 indices
typedef void(*callback_pair_t)(size_t, size_t);
// type for any function that takes a list_t by reference
typedef void(*callback_list_t)(list_t&);

inline size_t hammingweight(uint32_t n) {
   return __builtin_popcount(n);
}

void printsomestuff(list_t output) {
    for (size_t i = 0; i < output.size(); i++) {
        for (size_t j = 0; j < output[0].size(); j++) {
            cout << output[i][j], cout << '\n';
        }
    }
}

void NSS(const list_t& L, size_t t, callback_list_t f)  {

    list_t output;
    bitvec_t bob, anna;

    // go over all unique pairs 0 <= j < i < L.size()
    for (size_t i = 0; i < L.size(); ++i)    {
        for (size_t j = 0; j < i; ++j)    {
            // compute hamming weight of (L[i] ^ L[j])
            size_t w = 0;
            for (size_t k = 0; k < NW; ++k)
                w += hammingweight(L[i][k] ^ L[j][k]);
            // if below given threshold then put into output list
            if (w < t)
                output.emplace_back(bob,anna);
        }
        // periodically give outputlist back for further processing
        f(output); // assume it empties output
    }
}

int main() {
    bitvec_t mijn = [1,1];
    list_t test = [mijn,mijn];
    size_t thersh = 9298283798291;
    NSS(test, thersh, printsomestuff);
    return 0;

}
