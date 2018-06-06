#include <iostream>
#include <bitset>
using namespace std;




int do_some_loopjes( int bit_vec1, int bit_vec2)
{
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < i; j++)
        {
            cout << std::bitset<16>((i*j*bit_vec1)^(i*j*bit_vec2)), cout << '\n';
        }
    }
}

int main()
{
    int vec1, vec2;
    cout << "Hello, World!";
    cin >> vec1;
    cin >> vec2;
    cout << do_some_loopjes(vec1,vec2);
    return 0;
}

void NNS(L, t)  {
    for (int i = 0; i < L.size(); i++) {
        for (int j = 0; j < i; j++) {

        }
    }
}
