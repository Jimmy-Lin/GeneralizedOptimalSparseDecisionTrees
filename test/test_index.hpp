
#include "../src/index.hpp"

int test_index(void) {
    int failures = 0;

    std::vector< float > data{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    Index index(data);

    bitset mask(10);
    for (unsigned int i = 0; i < 10; i += 2) { mask[i] = 1; }

    // 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8 + 0.9 + 1.0
    failures += expect(5.5, index.sum(), "Test Index::sum() Failed.");

    // 0.1 + 0.3 + 0.5 + 0.7 + 0.9 
    failures += expect(2.5, index.sum(Bitmask(mask)), "Test Index::sum(mask) Failed.");

    std::vector< float > alpha{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    std::vector< bool > beta{true, false, true, false, true, false, true, false, true, false};
    Bitmask gamma = Bitmask::ones(10);
    blasvector u(alpha.size());
    blasvector v(beta.size());
    std::copy(alpha.begin(), alpha.end(), u.begin());
    std::copy(beta.begin(), beta.end(), v.begin());

    // float result = index.sum(gamma, u);
    // failures += expect(3.85, result, "Test Index::sum(u, v) Failed.");

    return failures;

}