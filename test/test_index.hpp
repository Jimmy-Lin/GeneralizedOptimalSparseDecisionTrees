
#include "../src/index.hpp"

int test_index(void) {
    int failures = 0;

    std::vector< std::vector< float > > data;
    for (unsigned int i = 0; i < 10; ++i) {
        std::vector< float > row;
        for (unsigned int j = 0; j < 10; ++j) {
            row.emplace_back(i * j * 0.01);
        }
        data.emplace_back(row);
    }
    Index index(data);

    unsigned int size = 10;

    {
        Bitmask mask(size); // 10101010101......
        for (unsigned int i = 0; i < size; i += 2) { mask.set(i, true); }

        std::stringstream context;
        context << "Test Bitmask Sum with Mask: " << mask.to_string();

        std::vector<float> accumulator(size, 0.0);
        std::vector<float> expectation(size, 0.0);

        for (unsigned int i = 0; i < 10; ++i) {
            if (!mask.get(i)) { continue; }
            for (unsigned int j = 0; j < 10; ++j) {
                expectation[j] += data[i][j];
            }
        }

        index.sum(mask, accumulator.data());

        for (unsigned int j = 0; j < 10; ++j) {
            std::stringstream specifier;
            specifier << "Index::sum element " << j << " is incorrect";
            failures += expect_fuzzy(expectation[j], accumulator[j], specifier.str(), context.str(), 10);
        }
    }

    {
        Bitmask mask(size); // 010101010101......
        for (unsigned int i = 1; i < size; i += 2) { mask.set(i, true); }

        std::stringstream context;
        context << "Test Bitmask Sum with Mask: " << mask.to_string();

        std::vector<float> accumulator(size, 0.0);
        std::vector<float> expectation(size, 0.0);

        for (unsigned int i = 0; i < 10; ++i) {
            if (!mask.get(i)) { continue; }
            for (unsigned int j = 0; j < 10; ++j) {
                expectation[j] += data[i][j];
            }
        }

        index.sum(mask, accumulator.data());

        for (unsigned int j = 0; j < 10; ++j) {
            std::stringstream specifier;
            specifier << "Index::sum element " << j << " is incorrect";
            failures += expect_fuzzy(expectation[j], accumulator[j], specifier.str(), context.str(), 10);
        }
    }

    {
        Bitmask mask(size); // 111111000000......
        for (unsigned int i = 0; i < size/2; i += 1) { mask.set(i, true); }

        std::stringstream context;
        context << "Test Bitmask Sum with Mask: " << mask.to_string();

        std::vector<float> accumulator(size, 0.0);
        std::vector<float> expectation(size, 0.0);

        for (unsigned int i = 0; i < 10; ++i) {
            if (!mask.get(i)) { continue; }
            for (unsigned int j = 0; j < 10; ++j) {
                expectation[j] += data[i][j];
            }
        }

        index.sum(mask, accumulator.data());

        for (unsigned int j = 0; j < 10; ++j) {
            std::stringstream specifier;
            specifier << "Index::sum element " << j << " is incorrect";
            failures += expect_fuzzy(expectation[j], accumulator[j], specifier.str(), context.str(), 10);
        }
    }

    {
        Bitmask mask(size); // 00000111111......
        for (unsigned int i = size/2; i < size; i += 1) { mask.set(i, true); }

        std::stringstream context;
        context << "Test Bitmask Sum with Mask: " << mask.to_string();

        std::vector<float> accumulator(size, 0.0);
        std::vector<float> expectation(size, 0.0);

        for (unsigned int i = 0; i < 10; ++i) {
            if (!mask.get(i)) { continue; }
            for (unsigned int j = 0; j < 10; ++j) {
                expectation[j] += data[i][j];
            }
        }

        index.sum(mask, accumulator.data());

        for (unsigned int j = 0; j < 10; ++j) {
            std::stringstream specifier;
            specifier << "Index::sum element " << j << " is incorrect";
            failures += expect_fuzzy(expectation[j], accumulator[j], specifier.str(), context.str(), 10);
        }
    }

    return failures;
}