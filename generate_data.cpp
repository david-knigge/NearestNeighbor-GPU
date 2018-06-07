#include <random>

using namespace std;

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
    for (size_t i = 0; i < n; ++n)  {
        for (size_t k = 0; k < NW; ++k)
            output[i][k] = mt();
    }
    // output list is given by reference, so nothing to return
}