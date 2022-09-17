#include "index.hpp"

Index::Index(void) {}

Index::Index(std::vector< std::vector< float > > const & src) {
    this -> size = src.size();
    this -> width = src.begin() -> size();
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, &number_of_blocks, &block_offset);
    this -> num_blocks = number_of_blocks;

    build_prefixes(src, this -> prefixes);

    this -> source.resize(this -> size * this -> width, 0.0);
    for (unsigned int i = 0;  i < this -> size; ++i) {
        for (unsigned int j = 0; j < this -> width; ++j) {
            this -> source[i * this -> width + j] = src.at(i).at(j);
        }
    }
}

Index::~Index(void) {}


void Index::precompute(void) {
}

void Index::build_prefixes(std::vector< std::vector< float > > const & src, std::vector< std::vector< float > > & prefixes) {
    // Compute a scan over the source vector, tracker[i] == sum(source[0..i-1])
    std::vector< float > init(this -> width, 0.0); // First entries for the prefix sum
    prefixes.emplace_back(init);
    for (unsigned int i = 0; i < this -> size; i++) {
        std::vector< float > const & source_row = src.at(i);
        std::vector< float > const & prefix_row = prefixes.at(i);
        std::vector< float > new_row;
        for (unsigned int j = 0; j < this -> width; ++j) {
            new_row.emplace_back(source_row.at(j) + prefix_row.at(j));
        }
        prefixes.emplace_back(new_row);
    }
}

void Index::sum(Bitmask const & indicator, float * accumulator) const {
    bit_sequential_sum(indicator, accumulator);
    float const epsilon = std::numeric_limits<float>::epsilon();
    for (unsigned int j = 0; j < this -> width; ++j) {
        if (accumulator[j] < epsilon) { accumulator[j] = 0.0; }
    }
}

void Index::bit_sequential_sum(Bitmask const & indicator, float * accumulator) const {
    unsigned int max = indicator.size();
    bool sign = true;
    unsigned int i = indicator.scan(0, true);;
    unsigned int j = indicator.scan(i, !sign);
    while (j <= max) {
        if (sign) {
            {
                std::vector< float > const & finish = this -> prefixes.at(j);
                for (int k = this -> width; --k >= 0;) { accumulator[k] += finish.at(k); }
            }
            {
                std::vector< float > const & start = this -> prefixes.at(i);
                for (int k = this -> width; --k >= 0;) { accumulator[k] -= start.at(k); }
            }
        }
        if (j == max) { break; }
        i = j;
        sign = !sign;
        j = indicator.scan(i, !sign);
    }
}

void Index::block_sequential_sum(bitblock * blocks, float * accumulator) const {
    unsigned int offset = 0;
    for (unsigned int i = 0; i < this -> num_blocks; ++i) {
        bitblock block = blocks[i];
        for (unsigned int range_index = 0; range_index < sizeof(bitblock) / sizeof(rangeblock); ++range_index) {
            unsigned int local_offset = offset + range_index * 8 * sizeof(rangeblock);
            unsigned int shift = range_index * 8 * sizeof(rangeblock);
            block_sequential_sum(0xffff & (block >> shift), local_offset, accumulator);
        }
        offset += 8 * sizeof(bitblock);
    }
}

void Index::block_sequential_sum(rangeblock block, unsigned int offset, float * accumulator) const {
    unsigned int local_offset = offset;
    bool positive = ((block) & 1) == 1;

    std::vector<codeblock> const & encoding = Bitmask::ranges[block];
    for (auto iterator = encoding.begin(); iterator != encoding.end(); ++iterator) {
        codeblock packed_code = * iterator;
        unsigned short code;

        for (unsigned int range_index = 0; range_index < Bitmask::ranges_per_code; ++range_index) {
            if (local_offset >= offset + 16 || local_offset >= this -> size) { break; }

            unsigned int shift = range_index * Bitmask::bits_per_range;
            code = (0xF & (packed_code >> shift)) + 1;

            if (positive) {
                std::vector< float > const & start = this -> prefixes.at(local_offset);
                std::vector< float > const & finish = this -> prefixes.at(local_offset + code);
                for (unsigned int j = 0; j < this -> width; ++j) { accumulator[j] += finish.at(j) - start.at(j); }
            }
            local_offset += code;
            positive = !positive;
        }
    }
}

std::string Index::to_string(void) const {
    std::stringstream stream;
    stream << "[";
    for (unsigned int i = 0; i < this -> size; ++i) {
        for (unsigned int j = 0; j < this -> width; ++j) {
            stream << this -> source[i * this -> width + j];
            stream << ",";
        }
        if (i+1 < this -> size) { stream << std::endl; }
    }
    stream << "]";
    return stream.str();
}

void Index::benchmark(void) const
{
    Bitmask indicator(this->size, true);
    for (unsigned int i = 0; i < this->size; ++i)
    {
        indicator.set(i, (i % 7 != 0));
    }
    bitblock *blocks = indicator.data();
    std::vector<float, tbb::scalable_allocator<float>> accumulator(this->width);
    unsigned int sample_size = 10000;

    float block_min = std::numeric_limits<float>::max();
    float block_max = -std::numeric_limits<float>::max();
    float block_avg;

    float bit_min = std::numeric_limits<float>::max();
    float bit_max = -std::numeric_limits<float>::max();
    float bit_avg;

    auto block_start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        block_sequential_sum(blocks, accumulator.data());
        auto finish = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);

        block_min = std::min((float)duration.count() / 1000, block_min);
        block_max = std::max((float)duration.count() / 1000, block_max);
    }
    auto block_finish = std::chrono::high_resolution_clock::now();
    block_avg = (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(block_finish - block_start).count()) / sample_size / 1000;

    auto bit_start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < sample_size; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        bit_sequential_sum(indicator, accumulator.data());
        auto finish = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start);

        bit_min = std::min((float)duration.count() / 1000, bit_min);
        bit_max = std::max((float)duration.count() / 1000, bit_max);
    }
    auto bit_finish = std::chrono::high_resolution_clock::now();
    bit_avg = (float)(std::chrono::duration_cast<std::chrono::nanoseconds>(bit_finish - bit_start).count()) / sample_size / 1000;

    std::cout << "Index Benchmark Results: " << std::endl;
    std::cout << "Block Sequential: " << std::endl;
    std::cout << "  Min: " << block_min << " ms" << std::endl;
    std::cout << "  Avg: " << block_avg << " ms" << std::endl;
    std::cout << "  Max: " << block_max << " ms" << std::endl;
    std::cout << "Bit Sequential: " << std::endl;
    std::cout << "  Min: " << bit_min << " ms" << std::endl;
    std::cout << "  Avg: " << bit_avg << " ms" << std::endl;
    std::cout << "  Max: " << bit_max << " ms" << std::endl;

    exit(1);
}
