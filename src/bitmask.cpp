#include "bitmask.hpp"
#include <cmath>
#include <cstring>

// ********************************
// ** Function Module Definition **
// ********************************

std::vector< std::vector<codeblock> > Bitmask::ranges = std::vector< std::vector<codeblock> >();
std::vector<size_t> Bitmask::hashes = std::vector<size_t>();
std::vector<char> Bitmask::counts = std::vector<char>();

// Pre-computed number of set bits for 4-bit sequences
// unsigned int Bitmask::bit_count[] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };

const bitblock Bitmask::bits_per_block = 8 * sizeof(bitblock);
const bitblock Bitmask::ranges_per_code = 8 * sizeof(codeblock) / log2((double)(8 * sizeof(rangeblock)));
const bitblock Bitmask::bits_per_range = log2((double)(8 * sizeof(rangeblock)));

tbb::scalable_allocator< bitblock > Bitmask::allocator = tbb::scalable_allocator< bitblock >();
bool Bitmask::integrity_check = true;
bool Bitmask::precomputed = false;

// @param blocks: the blocks containing bits
// @param size: the number of bits which are represented in blocks
// @modifies blocks will be set to have all bits be 1
void Bitmask::ones(bitblock * const blocks, unsigned int size) {
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    for (unsigned int i = 0; i < number_of_blocks; ++i) { blocks[i] = ~((bitblock)(0)); }
    Bitmask::clean(blocks, number_of_blocks, block_offset); 
}

// @param blocks: the blocks containing bits
// @param size: the number of bits which are represented in blocks
// @modifies blocks will be set to have all bits be 0
void Bitmask::zeros(bitblock * const blocks, unsigned int size) {
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    for (unsigned int i = 0; i < number_of_blocks; ++i) { blocks[i] = (bitblock)(0); }
    Bitmask::clean(blocks, number_of_blocks, block_offset);
}

void Bitmask::copy(bitblock * const blocks, bitblock * const other_blocks, unsigned int size) {
    if (blocks == other_blocks) { return; }
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);
    Bitmask::clean(other_blocks, number_of_blocks, block_offset);
    for (unsigned int i = 0; i < number_of_blocks; ++i) { other_blocks[i] = blocks[i]; }
}

void Bitmask::block_layout(unsigned int size, unsigned int * number_of_blocks, unsigned int * block_offset) {
    if (size == 0) {
        * number_of_blocks = 1;
    } else {
        * number_of_blocks = size / (Bitmask::bits_per_block) + (int)(size % Bitmask::bits_per_block != 0);
    }
    *block_offset = size % (Bitmask::bits_per_block);
}

void Bitmask::clean(bitblock * const blocks, unsigned int number_of_blocks, unsigned int offset) {
    if (offset == 0) { return; }
    bitblock mask = ~((bitblock)(0)) >> (Bitmask::bits_per_block - offset);
    blocks[number_of_blocks - 1] = blocks[number_of_blocks - 1] & mask;
}

unsigned int Bitmask::count(bitblock * const blocks, unsigned int size) {
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);
    return mpn_popcount(blocks, number_of_blocks);
}

// @note this returns the number of contiguous sequences of 1's
unsigned int Bitmask::words(bitblock * const blocks, unsigned int size) {
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);

    bool sign = Bitmask::get(blocks, size, 0);
    unsigned int i = 0;
    unsigned int j = Bitmask::scan(blocks, size, i, !sign);
    unsigned int words = 0;
    while (j <= size) {
        if (sign) { ++words; }
        if (j == size) { break; }
        i = j;
        sign = !sign;
        j = Bitmask::scan(blocks, size, i, !sign);
    }
    return words;
}

void Bitmask::bit_and(bitblock * const blocks, bitblock * other_blocks, unsigned int size, bool flip) {
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);

    if (!flip) {
        // Special Offload to GMP Implementation
        mpn_and_n(other_blocks, other_blocks, blocks, number_of_blocks);
    } else {
        // Special Offload to GMP Implementation
        mpn_nior_n(other_blocks, other_blocks, other_blocks, number_of_blocks);
        mpn_nior_n(other_blocks, other_blocks, blocks, number_of_blocks);
    }
}

void Bitmask::bit_or(bitblock * blocks, bitblock * other_blocks, unsigned int size, bool flip) {
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);

    if (!flip) {
        // Special Offload to GMP Implementation
        mpn_ior_n(other_blocks, other_blocks, blocks, number_of_blocks);
    } else {
        // Special Offload to GMP Implementation
        mpn_nand_n(other_blocks, other_blocks, other_blocks, number_of_blocks);
        mpn_nand_n(other_blocks, other_blocks, blocks, number_of_blocks);
    }
}

void Bitmask::bit_xor(bitblock * const blocks, bitblock * other_blocks, unsigned int size, bool flip) {
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);

    if (!flip) {
        // Special Offload to GMP Implementation
        mpn_xor_n(other_blocks, other_blocks, blocks, number_of_blocks);
    } else {
        // Special Offload to GMP Implementation
        mpn_xnor_n(other_blocks, other_blocks, blocks, number_of_blocks);
    }
}

bool Bitmask::equals(bitblock * const blocks, bitblock * const other_blocks, unsigned int size, bool flip) {
    if (blocks == other_blocks) { return true; }
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);
    Bitmask::clean(other_blocks, number_of_blocks, block_offset);

    if (!flip) {
        return mpn_cmp(blocks, other_blocks, number_of_blocks) == 0;
    } else {
        mpn_nand_n(blocks, blocks, blocks, number_of_blocks);
        Bitmask::clean(blocks, number_of_blocks, block_offset);
        bool equals = mpn_cmp(blocks, other_blocks, number_of_blocks) == 0;
        mpn_nand_n(blocks, blocks, blocks, number_of_blocks);
        Bitmask::clean(blocks, number_of_blocks, block_offset);
        return equals;
    }
}

int Bitmask::compare(bitblock * const left, bitblock * const right, unsigned int size) {
    if (left == right) { return false; }
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(left, number_of_blocks, block_offset);
    Bitmask::clean(right, number_of_blocks, block_offset);
    return mpn_cmp(left, right, number_of_blocks);
}

bool Bitmask::less_than(bitblock * const left, bitblock * const right, unsigned int size) {
    return Bitmask::compare(left, right, size) < 0;
}

bool Bitmask::greater_than(bitblock * const left, bitblock * const right, unsigned int size) {
    return Bitmask::compare(left, right, size) > 0;
}

size_t Bitmask::hash(bitblock * const blocks, unsigned int size) {
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);

    bool sign = Bitmask::get(blocks, size, 0);
    unsigned int i = 0;
    unsigned int j = Bitmask::scan(blocks, size, i, !sign);
    size_t seed = 1 * sign;
    while (j <= size) {
        seed ^= j - i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        if (j == size) { break; }
        i = j;
        sign = !sign;
        j = Bitmask::scan(blocks, size, i, !sign);
    }
    return seed;
}

unsigned int Bitmask::get(bitblock * const blocks, unsigned int size, unsigned int index) {
    if (Bitmask::integrity_check && (index < 0 || index >= size)) {
        std::stringstream reason;
        reason << "Index " << index << " is outside the valid range [" << 0 << "," << size - 1 << "].";
        throw IntegrityViolation("Bitmask::get", reason.str());
    }
    unsigned int block_index = index / Bitmask::bits_per_block;
    unsigned int bit_index = index % Bitmask::bits_per_block;

    bitblock block = blocks[block_index];
    return (block >> bit_index) & 1;
}

void Bitmask::set(bitblock * const blocks, unsigned int size, unsigned int index, bool value) {
    if (Bitmask::integrity_check && (index < 0 || index >= size)) { 
        std::stringstream reason;
        reason << "Index " << index << " is outside the valid range [" << 0 << "," << size - 1 << "].";
        throw IntegrityViolation("Bitmask::get", reason.str());
     }
    unsigned int block_index = index / Bitmask::bits_per_block;
    unsigned int bit_index = index % Bitmask::bits_per_block;

    bitblock mask = (bitblock)(1) << bit_index;
    if (value) {
        blocks[block_index] = blocks[block_index] | mask;
    } else {
        blocks[block_index] = blocks[block_index] & ~mask;
    }
}

int Bitmask::scan(bitblock * const blocks, int size, int start, bool value) {
    if (start >= size) { return size; }
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);

    unsigned int block_index = start / Bitmask::bits_per_block;
    if (block_index >= number_of_blocks) { return size; }
    if (value) {
        bitblock skip_block = (bitblock)(0);
        bitblock mask_block = ~((bitblock)(0)) << (start % Bitmask::bits_per_block);
        bitblock block = blocks[block_index] & mask_block; // clear lower bits to ignore them
        while (block == skip_block) {
            ++block_index;
            if (block_index >= number_of_blocks) { return size; }
            block = blocks[block_index];
        }
        int bit_index = mpn_scan1(& block, 0);
        return block_index * Bitmask::bits_per_block + bit_index;
    } else {
        bitblock skip_block = ~((bitblock)(0));
        bitblock mask_block = ((bitblock)(1) << (start % Bitmask::bits_per_block)) - (bitblock)(1);
        bitblock block = blocks[block_index] | mask_block; // Set lower bits to ignore them
        while (block == skip_block) {
            ++block_index;
            if (block_index >= number_of_blocks) { return size; }
            block = blocks[block_index];
        }
        int bit_index = mpn_scan0(& block, 0);
        return block_index * Bitmask::bits_per_block + bit_index;
    }
}


int Bitmask::rscan(bitblock * const blocks, int size, int start, bool value) {
    if (start < 0) { return -1; }
    unsigned int number_of_blocks, block_offset;
    Bitmask::block_layout(size, & number_of_blocks, & block_offset);
    Bitmask::clean(blocks, number_of_blocks, block_offset);

    int block_index = start / Bitmask::bits_per_block;
    if (block_index < 0) { return -1; }
    if (value) {
        bitblock skip_block = (bitblock)(0);
        bitblock mask_block = ~((bitblock)(0)) >> (Bitmask::bits_per_block - 1 - (start % Bitmask::bits_per_block));
        bitblock block = blocks[block_index] & mask_block; // clear lower bits to ignore them

        while (block == skip_block) {
            --block_index;
            if (block_index < 0) { return -1; }
            block = blocks[block_index];
        }

        unsigned int count = sizeof(block) * 8 - 1;
        bitblock reverse_block = block;
        block >>= 1;
        while (block) {
            reverse_block <<= 1;
            reverse_block |= block & 1;
            block >>= 1;
            count--;
        }
        reverse_block <<= count;

        int bit_index = mpn_scan1(& reverse_block, 0);
        return (block_index + 1) * Bitmask::bits_per_block - 1 - bit_index;
    } else {
        bitblock skip_block = ~((bitblock)(0));
        bitblock mask_block = ~((bitblock)(0)) >> (Bitmask::bits_per_block - 1 - (start % Bitmask::bits_per_block));
        mask_block = ~mask_block;
        bitblock block = blocks[block_index] | mask_block; // Set lower bits to ignore them

        while (block == skip_block) {
            --block_index;
            if (block_index < 0) { return -1; }
            block = blocks[block_index];
        }

        unsigned int count = sizeof(block) * 8 - 1;
        bitblock reverse_block = block;
        block >>= 1;
        while (block) {
            reverse_block <<= 1;
            reverse_block |= block & 1;
            block >>= 1;
            count--;
        }
        reverse_block <<= count;

        int bit_index = mpn_scan0(& reverse_block, 0);
        return (block_index + 1) * Bitmask::bits_per_block - 1 - bit_index;
    }
}

bool Bitmask::scan_range(bool value, int & begin, int & end) const {
    if (begin >= this -> _size) { return false; }
    begin = this -> scan(begin, value);
    if (begin >= this -> _size) { return false; }
    end = this -> scan(begin, !value);
    return true;
}

bool Bitmask::rscan_range(bool value, int & begin, int & end) const {
    if (begin < 0) { return false; }
    begin = this -> rscan(begin, value);
    if (begin < 0) { return false; }
    end = this -> rscan(begin, !value);
    return true;
}

std::string Bitmask::to_string(bitblock * const blocks, unsigned int size, bool reverse) {
    std::string bitstring;
    bitstring.resize(size);
    char zero = '0';
    char one = '1';
    if (reverse) { // Copy the bits
        for (unsigned int i = 0; i < size; ++i) { bitstring[i] = Bitmask::get(blocks, size, size - 1 - i) ? one : zero; }
    } else {
        for (unsigned int i = 0; i < size; ++i) { bitstring[i] = Bitmask::get(blocks, size, i) ? one : zero; }
    }
    return bitstring;
};

void Bitmask::precompute(void) {
    if (Bitmask::precomputed) { return; }
    Bitmask::precomputed = true;
    std::map< rangeblock, std::vector<char> > collection;
    char block_size = 8 * sizeof(rangeblock);
    rangeblock max = ~((rangeblock)0);
    for (rangeblock key = 0; key <= max; ++key) {
        std::vector<char> code;
        if (key  == (rangeblock)0) {
            code.emplace_back((char)(-block_size));
        } else if (key == (rangeblock)1) {
            code.emplace_back((char)1);
            code.emplace_back((char)(-block_size+1));
        } else {
            unsigned int prefix_length = std::floor(log2((double)key));
            unsigned int suffix_length = block_size - prefix_length;
            unsigned int prefix_mask = ~(~0 << prefix_length);
            rangeblock prefix_key = key & prefix_mask;
            unsigned int prior = (key >> (prefix_length-1)) & 1;
            unsigned int bit = (key >> prefix_length) & 1;
            std::vector<char> const & prefix = collection.at(prefix_key);
            code = prefix;
            if (bit == 1) {
                if (prior == 0) {         
                    char suffix = code.at(code.size()-1);
                    code[code.size() - 1] = suffix + suffix_length;
                    code.emplace_back(1);
                    if (1 - suffix_length != 0) {
                        code.emplace_back(1 - suffix_length);
                    }
                } else if (prior == 1) {
                    code[code.size() - 2] += 1;
                    code[code.size() - 1] += 1;
                    if (code[code.size() - 1] == 0) {
                        code.pop_back();
                    }
                }
            }
        }
        collection.emplace(key, code);
        if (key == max) { break; }
    }

    for (auto iterator = collection.begin(); iterator != collection.end(); iterator++) {
        std::vector<char> const & encoding = iterator -> second;
        std::vector<codeblock> packed_encoding;
        unsigned int counter = 0;
        codeblock packed_code = 0;
        unsigned int offset = 0;
        for (auto subiterator = encoding.begin(); subiterator != encoding.end(); ++subiterator) {
            short code = * subiterator; // some char representing anywhere from 1 to 16 bits
            if (code > 0) { counter += code; }
            if (offset == Bitmask::ranges_per_code) {
                packed_encoding.emplace_back(packed_code);
                packed_code = 0;
                offset = 0;
            }
            unsigned int shift = offset * Bitmask::bits_per_range;
            packed_code = packed_code | ((codeblock)(std::abs(code) - 1) << shift);
            ++offset;
        }
        if (offset > 0) {
            packed_encoding.emplace_back(packed_code);
        }
        Bitmask::ranges.emplace_back(packed_encoding);
        Bitmask::counts.emplace_back(counter);

        bool leading_sign = ((iterator -> first) & 1) == 1;
        size_t seed = leading_sign;
        std::vector<codeblock> const & codes = packed_encoding;
        for (auto code_iterator = codes.begin(); code_iterator != codes.end(); ++code_iterator) {
            seed ^=  * code_iterator + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            if ((int)(* code_iterator) > 0) { counter += * code_iterator; }
        }
        Bitmask::hashes.emplace_back(seed);
    }
}

// **********************
// ** Class Definition **
// **********************

Bitmask::Bitmask(void) {
    set_depth_budget(0);
}

Bitmask::Bitmask(unsigned int size, bool filler, bitblock * local_buffer, unsigned char depth_budget) {
    initialize(size, local_buffer);

    if (filler) { fill(); } else { clear(); }

    Bitmask::clean(this -> content, this -> _used_blocks, this -> _offset);

    this->set_depth_budget(depth_budget);
}

Bitmask::Bitmask(bitblock * source_blocks, unsigned int size, bitblock * local_buffer, unsigned char depth_budget) {
    if (Bitmask::integrity_check && source_blocks == NULL) {
        std::stringstream reason;
        reason << "Attempt to construct Bitmask from null source";
        throw IntegrityViolation("Bitmask::Bitmask", reason.str());
    }
    initialize(size, local_buffer);

    memcpy(this -> content, source_blocks, this -> _used_blocks * sizeof(bitblock));

    Bitmask::clean(this -> content, this -> _used_blocks, this -> _offset);

    this->set_depth_budget(depth_budget);
}

//Bitmask::Bitmask(dynamic_bitset const & source, bitblock * local_buffer, unsigned char depth_budget) {
//    initialize(source.size(), local_buffer);
//
//    // Initialize content using the blocks of this bitset
//    std::vector< bitblock > source_blocks;
//    source_blocks.resize(source.num_blocks());
//    boost::to_block_range(source, source_blocks.begin());
//
//    memcpy(this -> content, source_blocks.data(), this -> _used_blocks * sizeof(bitblock));
//    Bitmask::clean(this -> content, this -> _used_blocks, this -> _offset);
//
//    this->set_depth_budget(depth_budget);
//}

Bitmask::Bitmask(Bitmask const & source, bitblock * local_buffer) {
    if (source._size == 0) { return; }
    if (Bitmask::integrity_check && !source.valid()) {
        std::stringstream reason;
        reason << "Attempt to construct Bitmask from null source";
        throw IntegrityViolation("Bitmask::Bitmask", reason.str());
    }
    initialize(source.size(), local_buffer);
    memcpy(this -> content, source.data(), this -> _used_blocks * sizeof(bitblock));
    Bitmask::clean(this->content, this->_used_blocks, this->_offset);
    
    this -> set_depth_budget(source.get_depth_budget());
}

Bitmask::~Bitmask(void) {
    if (this -> shallow == false && valid()) {
        Bitmask::allocator.deallocate(this -> content, this -> _max_blocks);
    }
}

void Bitmask::initialize(unsigned int size, bitblock * local_buffer) {
    this -> _size = size;
    unsigned int num_blocks;
    Bitmask::block_layout(this -> _size, & num_blocks, & (this -> _offset));
    this -> _used_blocks = this -> _max_blocks = num_blocks;
    if (local_buffer == NULL) {
        this -> content = (bitblock *) Bitmask::allocator.allocate(this -> _max_blocks);
    } else {
        this -> content = local_buffer;
        this -> shallow = true;
    }
    Bitmask::clean(this -> content, this -> _used_blocks, this -> _offset);
}

void Bitmask::resize(unsigned int new_size) {

    if (this -> _size == new_size) { return; }
    if (this -> content == NULL) { 
        initialize(new_size);
    } else if (Bitmask::integrity_check && new_size > (this -> capacity())) {
        std::cout << "Resize: " << new_size << ", Capacity: " << this -> capacity() << std::endl;
        std::stringstream reason;
        reason << "Attempt to resize beyond allocated capacity";
        throw IntegrityViolation("Bitmask::resize", reason.str());
    }
    this -> _size = new_size;
    Bitmask::block_layout(new_size, & (this -> _used_blocks), & (this -> _offset));
    Bitmask::clean(this -> content, this -> _used_blocks, this -> _offset);
}

void Bitmask::copy_to(bitblock * dest_blocks) const {
    if (this -> _size == 0) { return; }
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Attempt to copy from null source";
        throw IntegrityViolation("Bitmask::copy_to", reason.str());
    }
    if (Bitmask::integrity_check && dest_blocks == NULL) {
        std::stringstream reason;
        reason << "Attempt to copy to null destination";
        throw IntegrityViolation("Bitmask::copy_to", reason.str());
    }
    Bitmask::copy(this -> content, dest_blocks, this -> _size);
}

void Bitmask::copy_from(bitblock * src_blocks) {
    if (Bitmask::integrity_check && src_blocks == NULL) {
        std::stringstream reason;
        reason << "Attempt to copy from null source";
        throw IntegrityViolation("Bitmask::copy_from", reason.str());
    }
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Attempt to copy to null destination";
        throw IntegrityViolation("Bitmask::copy_from", reason.str());
    }
    Bitmask::copy(src_blocks, this -> content, this -> _size);
}

Bitmask & Bitmask::operator=(Bitmask const & other) {
    if (other.size() == 0) { return * this; }
    if (this -> content == NULL) { initialize(other.size()); } // resize this instance to match
    if (this -> _size != other.size()) { resize(other.size()); } // resize this instance to match
    bitblock * blocks = this -> content;
    bitblock * other_blocks = other.content;
    memcpy(blocks, other_blocks, this -> _used_blocks * sizeof(bitblock));
    this -> set_depth_budget(other.get_depth_budget());
    return * this;
}

bitblock * Bitmask::data(void) const {
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Accessing invalid data";
        throw IntegrityViolation("Bitmask::data", reason.str());
    }
    return this -> content;
}

unsigned int Bitmask::operator[](unsigned int index) const {
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Accessing invalid data";
        throw IntegrityViolation("Bitmask::operator[]", reason.str());
    }
    return get(index);
}

unsigned int Bitmask::get(unsigned int index) const {
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Accessing invalid data";
        throw IntegrityViolation("Bitmask::get", reason.str());
    }
    bitblock * blocks = this -> content;
    unsigned int block_index = index / Bitmask::bits_per_block;
    unsigned int bit_index = index % Bitmask::bits_per_block;
    bitblock block = blocks[block_index];
    return (int)((block >> bit_index) % 2);
}

void Bitmask::set(unsigned int index, bool value) {
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Accessing invalid data";
        throw IntegrityViolation("Bitmask::set", reason.str());
    }
    bitblock * blocks = this -> content;
    unsigned int block_index = index / Bitmask::bits_per_block;
    unsigned int bit_index = index % Bitmask::bits_per_block;
    bitblock mask = (bitblock)(1) << bit_index;
    if (value) {
        blocks[block_index] = blocks[block_index] | mask;
    } else {
        blocks[block_index] = blocks[block_index] & ~mask;
    }
}

unsigned char Bitmask::get_depth_budget() const {
    return this->depth_budget;
}

void Bitmask::set_depth_budget(unsigned char depth_budget) {
    this->depth_budget = depth_budget;
}

unsigned int Bitmask::size(void) const { return this -> _size; }

unsigned int Bitmask::capacity(void) const { return this -> _max_blocks * Bitmask::bits_per_block; }


unsigned int Bitmask::count(void) const {
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Accessing invalid data";
        throw IntegrityViolation("Bitmask::count", reason.str());
    }
    return mpn_popcount(this -> content, this -> _used_blocks);
}

bool Bitmask::empty(void) const {
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Accessing invalid data";
        throw IntegrityViolation("Bitmask::empty", reason.str());
    }
    return mpn_zero_p(this -> content, this -> _used_blocks);
}

bool Bitmask::full(void) const {
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Accessing invalid data";
        throw IntegrityViolation("Bitmask::full", reason.str());
    }
    return this -> count() == this -> size();
}

int Bitmask::scan(int start, bool value) const {
    if (start >= size()) { return size(); }
    bitblock * blocks = this -> content;
    unsigned int block_index = start / Bitmask::bits_per_block;
    if (block_index >= this -> _used_blocks) { return size(); }
    if (value) {
        bitblock skip_block = (bitblock)(0);
        bitblock mask_block = ~((bitblock)(0)) << (start % Bitmask::bits_per_block);
        bitblock block = blocks[block_index] & mask_block; // clear lower bits to ignore them

        while (block == skip_block) {
            ++block_index;
            if (block_index >= this -> _used_blocks) { return size(); }
            block = blocks[block_index];
        }
        int bit_index = mpn_scan1(& block, 0);
        return block_index * Bitmask::bits_per_block + bit_index;
    } else {
        bitblock skip_block = ~((bitblock)(0));
        bitblock mask_block = ((bitblock)(1) << (start % Bitmask::bits_per_block)) - (bitblock)(1);
        bitblock block = blocks[block_index] | mask_block; // Set lower bits to ignore them

        while (block == skip_block) {
            ++block_index;
            if (block_index >= this -> _used_blocks) { return size(); }
            block = blocks[block_index];
        }
        int bit_index = mpn_scan0(& block, 0);
        return block_index * Bitmask::bits_per_block + bit_index;
    }
}

int Bitmask::rscan(int start, bool value) const {
    if (start < 0) { return -1; }
    bitblock * blocks = this -> content;
    int block_index = start / Bitmask::bits_per_block;
    if (block_index < 0) { return -1; }
    if (value) {
        bitblock skip_block = (bitblock)(0);
        bitblock mask_block = ~((bitblock)(0)) >> (Bitmask::bits_per_block - 1 - (start % Bitmask::bits_per_block));
        bitblock block = blocks[block_index] & mask_block; // clear lower bits to ignore them

        while (block == skip_block) {
            --block_index;
            if (block_index < 0) { return -1; }
            block = blocks[block_index];
        }

        unsigned int count = sizeof(block) * 8 - 1;
        bitblock reverse_block = block;
        block >>= 1;
        while (block) {
            reverse_block <<= 1;
            reverse_block |= block & 1;
            block >>= 1;
            count--;
        }
        reverse_block <<= count;

        int bit_index = mpn_scan1(& reverse_block, 0);
        return (block_index + 1) * Bitmask::bits_per_block - 1 - bit_index;
    } else {
        bitblock skip_block = ~((bitblock)(0));
        bitblock mask_block = ~((bitblock)(0)) >> (Bitmask::bits_per_block - 1 - (start % Bitmask::bits_per_block));
        mask_block = ~mask_block;
        bitblock block = blocks[block_index] | mask_block; // Set lower bits to ignore them

        while (block == skip_block) {
            --block_index;
            if (block_index < 0) { return -1; }
            block = blocks[block_index];
        }

        unsigned int count = sizeof(block) * 8 - 1;
        bitblock reverse_block = block;
        block >>= 1;
        while (block) {
            reverse_block <<= 1;
            reverse_block |= block & 1;
            block >>= 1;
            count--;
        }
        reverse_block <<= count;

        int bit_index = mpn_scan0(& reverse_block, 0);
        return (block_index + 1) * Bitmask::bits_per_block - 1 - bit_index;
    }
}

unsigned int Bitmask::words(void) const {
    if (this -> _size == 0) { return 0; }
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Accessing invalid data";
        throw IntegrityViolation("Bitmask::words", reason.str());
    }

    unsigned int max = size();
    bool sign = get(0);
    unsigned int i = 0;
    unsigned int j = scan(i, !sign);
    unsigned int words = 0;
    while (j <= max) {
        if (sign) { ++words; }
        if (j == max) { break; }
        i = j;
        sign = !sign;
        j = scan(i, !sign);
    }
    return words;
}

void Bitmask::bit_and(bitblock * other_blocks, bool flip) const {
    if (Bitmask::integrity_check && (!valid() || other_blocks == NULL)) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::bit_and", reason.str());
    }
    Bitmask::bit_and(content, other_blocks, _size, flip);
}

void Bitmask::bit_and(Bitmask const & other, bool flip) const {
    if (this -> _size == 0 && other._size == 0) { return; }
    if (Bitmask::integrity_check && (!valid() || !other.valid())) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::bit_and", reason.str());
    }

    bitblock * blocks = this -> content;
    bitblock * other_blocks = other.content;
    unsigned int block_count = std::min(this -> _used_blocks, other._used_blocks);

    if (!flip) {
        // Special Offload to GMP Implementation
        mpn_and_n(other_blocks, blocks, other_blocks, block_count);
    } else {
        // Special Offload to GMP Implementation
        mpn_nior_n(other_blocks, other_blocks, other_blocks, block_count);
        mpn_nior_n(other_blocks, blocks, other_blocks, block_count);
    }
};

void Bitmask::bit_or(bitblock * other_blocks, bool flip) const {
    if (Bitmask::integrity_check && (!valid() || other_blocks == NULL)) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::bit_or", reason.str());
    }
    Bitmask::bit_or(content, other_blocks, _size, flip);
}

void Bitmask::bit_or(Bitmask const & other, bool flip) const {
    if (this -> _size == 0 && other._size == 0) { return; }
    if (Bitmask::integrity_check && (!valid() || !other.valid())) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::bit_or", reason.str());
    }
    bitblock * blocks = this -> content;
    bitblock * other_blocks = other.content;
    unsigned int block_count = std::min(this -> _used_blocks, other._used_blocks);

    if (!flip) {
        // Special Offload to GMP Implementation
        mpn_ior_n(other_blocks, blocks, other_blocks, block_count);
    } else {
        // Special Offload to GMP Implementation
        mpn_nand_n(other_blocks, other_blocks, other_blocks, block_count);
        mpn_nand_n(other_blocks, blocks, other_blocks, block_count);
    }
};

void Bitmask::bit_xor(bitblock * other_blocks, bool flip) const {
    if (Bitmask::integrity_check && (!valid() || other_blocks == NULL)) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::bit_xor", reason.str());
    }
    Bitmask::bit_xor(content, other_blocks, _size, flip);
}

void Bitmask::bit_xor(Bitmask const & other, bool flip) const {
    if (this -> _size == 0 && other._size == 0) { return; }
    if (Bitmask::integrity_check && (!valid() || !other.valid())) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::bit_xor", reason.str());
    }
    bitblock * blocks = this -> content;
    bitblock * other_blocks = other.content;
    unsigned int block_count = std::min(this -> _used_blocks, other._used_blocks);

    if (!flip) {
        // Special Offload to GMP Implementation
        mpn_xor_n(other_blocks, blocks, other_blocks, block_count);
    } else {
        // Special Offload to GMP Implementation
        mpn_xnor_n(other_blocks, blocks, other_blocks, block_count);
    }
};

void Bitmask::clear(void) {
    if (this -> _size == 0) { return; }
    bitblock * blocks = this -> content;
    for (unsigned int i = 0; i < this -> _used_blocks; ++i) {
        blocks[i] = (bitblock)(0);
    }
}

void Bitmask::fill(void) {
    if (this -> _size == 0) { return; }
    bitblock * blocks = this -> content;
    for (unsigned int i = 0; i < this -> _used_blocks; ++i) {
        blocks[i] = ~((bitblock)(0));
    }
    Bitmask::clean(this -> content, this -> _used_blocks, this -> _offset);
}

bool Bitmask::operator==(bitblock * other) const {
    if (Bitmask::integrity_check && (!valid() || other == NULL)) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::operator==", reason.str());
    }
    return Bitmask::equals(this -> content, other, this -> _size);
}
bool Bitmask::operator==(Bitmask const & other) const {
    if (this -> _size == 0 && other._size == 0) { return true; }
    if (Bitmask::integrity_check && (!valid() || !other.valid())) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::operator==", reason.str());
    }
    if (size() != other.size()) { return false; }
    if (this->get_depth_budget() != other.get_depth_budget()) { 
        return false;
    }
    return (mpn_cmp(this -> content, other.data(), this -> _used_blocks) == 0);
}

bool Bitmask::operator<(Bitmask const & other) const {
    if (Bitmask::integrity_check && (!valid() || !other.valid())) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::operator<", reason.str());
    }
    return Bitmask::less_than(this -> content, other.data(), this -> _size) ||
     (mpn_cmp(this -> content, other.data(), this -> _used_blocks) == 0 && this->get_depth_budget() < other.get_depth_budget());
}

bool Bitmask::operator>(Bitmask const & other) const {
    if (Bitmask::integrity_check && (!valid() || !other.valid())) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::operator>", reason.str());
    }
    return Bitmask::greater_than(this -> content, other.data(), this -> _size) ||
     (mpn_cmp(this -> content, other.data(), this -> _used_blocks) == 0 && this->get_depth_budget() > other.get_depth_budget());
}

bool Bitmask::operator!=(Bitmask const & other) const {
    if (this -> _size == 0 && other._size == 0) { return false; }
    if (Bitmask::integrity_check && (!valid() || !other.valid())) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::operator==", reason.str());
    }
    return !(* this == other);
}

bool Bitmask::operator<=(Bitmask const & other) const { return !(* this > other); }

bool Bitmask::operator>=(Bitmask const & other) const { return !(* this < other); }

// TODO: incorporate depth in hash
size_t Bitmask::hash(bool bitwise) const {
    size_t seed = this -> _size;
    if (this -> _size == 0) { return seed; }
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Operating with invalid data";
        throw IntegrityViolation("Bitmask::hash", reason.str());
    }
    // unsigned int max = size();
    // bool sign = get(0);
    // unsigned int i = 0;
    // unsigned int j = scan(i, !sign);
    // size_t seed = 1 * sign;
    // while (j <= max) {
    //     seed ^= j - i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    //     if (j == max) { break; }
    //     i = j;
    //     sign = !sign;
    //     j = scan(i, !sign);
    // }
    bitblock * blocks = this -> content;
    for (unsigned int i = 0; i < this -> _used_blocks; ++i) {
        seed ^= blocks[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

std::string Bitmask::to_string(bool reverse) const {
    if (this -> _size == 0) { return ""; }
    if (Bitmask::integrity_check && !valid()) {
        std::stringstream reason;
        reason << "Rendering with invalid data";
        throw IntegrityViolation("Bitmask::to_string", reason.str());
    }
    return Bitmask::to_string(this -> content, this -> _size);
}

bool Bitmask::valid(void) const {
    return this -> content != NULL;
}


void Bitmask::benchmark(unsigned int size) {
    unsigned int trials = 100;
    unsigned int samples = 1000;
    unsigned int length = size;
    unsigned int blocks, offset;
    Bitmask::block_layout(length, & blocks, & offset);
    unsigned int bytes = blocks * Bitmask::bits_per_block / 8;
    // unsigned int nails = Bitmask::bits_per_block - offset - 1;
    unsigned int nails = 0;

    float custom_copy = 0.0;
    float custom_compare = 0.0;
    float custom_count = 0.0;
    float custom_and = 0.0;
    float custom_or = 0.0;
    float custom_xor = 0.0;
    float custom_hash = 0.0;
    float custom_iterate = 0.0;

    float std_copy = 0.0;
    float std_compare = 0.0;

    float gmp_copy = 0.0;
    float gmp_compare = 0.0;
    float gmp_count = 0.0;
    float gmp_and = 0.0;
    float gmp_or = 0.0;
    float gmp_xor = 0.0;
    float gmp_hash = 0.0;
    float gmp_iterate = 0.0;

    std::cout << "Benchmarking Memory Copy..." << std::endl;
    for (unsigned int i = 10; i < trials; ++i) {
        Bitmask expectation(length);
        for (unsigned int j = 0; j < length; ++j) { expectation.set(j, j % 100 < i); }
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, (j+i) % 100 < i);
            }

            // Memory Copy: Custom
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                beta = alpha;
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            custom_copy += duration;
            if (beta != expectation) {
                std::cout << "Custom copy is incorrect." << std::endl;
                std::cout << "Expected: " << expectation.to_string() << std::endl;
                std::cout << "Got:      " << beta.to_string() << std::endl;
                exit(1);
            }
        }
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, (j+i) % 100 < i);
            }

            // Memory Copy: STD
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                memcpy(beta.data(), alpha.data(), bytes);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            std_copy += duration;
            if (beta != expectation) { 
                std::cout << "STD copy is incorrect." << std::endl;
                std::cout << "Expected: " << expectation.to_string() << std::endl;
                std::cout << "Got:      " << beta.to_string() << std::endl;
                exit(1);
            }
        }
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, (j+i) % 100 < i);
            }
            // Translation to mpz objects
            mpz_t mpz_alpha;
            mpz_t mpz_beta;
            mpz_init2(mpz_alpha, length);
            mpz_init2(mpz_beta, length);
            mpz_import(mpz_alpha, blocks, -1, sizeof(bitblock), -1, nails, alpha.data());
            mpz_import(mpz_beta, blocks, -1, sizeof(bitblock), -1, nails, beta.data());

            // Memory Copy: GMP
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                mpz_import(mpz_beta, blocks, -1, sizeof(bitblock), -1, nails, alpha.data());
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            gmp_copy += duration;

            // Clear necessary since mpz_export does not copy zero words (it assumes the destination memory has be cleared)
            alpha.clear();
            beta.clear();
            // Translation from mpz objects
            mpz_export(alpha.data(), NULL, -1, sizeof(bitblock), -1, nails, mpz_alpha);
            mpz_export(beta.data(), NULL, -1, sizeof(bitblock), -1, nails, mpz_beta);
            
            if (beta != expectation) { 
                std::cout << "GMP copy is incorrect." << std::endl;
                std::cout << "Expected: " << expectation.to_string() << std::endl;
                std::cout << "Got:      " << beta.to_string() << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Results:" << std::endl;
    std::cout << "  Custom Copy Average Runtime: " << (float)custom_copy / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  STD Copy Average Runtime: " << (float)std_copy / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  GMP Copy Average Runtime: " << (float)gmp_copy / (float)(trials * samples) << " ns" << std::endl;

    std::cout << "Benchmarking Memory Compare..." << std::endl;
    for (unsigned int i = 10; i < trials; ++i) {
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, j % 100 < i);
            }

            // Memory Compare: Custom
            auto start = std::chrono::high_resolution_clock::now();
            bool eq;
            for (unsigned int k = 0; k < samples; ++k) {
                eq = alpha == beta;
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            custom_compare += duration;
            if (!eq) {
                std::cout << "Custom compare is incorrect." << std::endl;
                exit(1);
            }
        }
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, j % 100 < i);
            }

            // Memory Compare: STD
            auto start = std::chrono::high_resolution_clock::now();
            int eq;
            for (unsigned int k = 0; k < samples; ++k) {
                eq = memcmp(alpha.data(), beta.data(), bytes);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            std_compare += duration;
            if (eq != 0) { 
                std::cout << "STD compare is incorrect." << std::endl;
                exit(1);
            }
        }
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, j % 100 < i);
            }
            // Translation to mpz objects
            mpz_t mpz_alpha;
            mpz_t mpz_beta;
            mpz_init2(mpz_alpha, length);
            mpz_init2(mpz_beta, length);
            mpz_import(mpz_alpha, blocks, -1, sizeof(bitblock), -1, nails, alpha.data());
            mpz_import(mpz_beta, blocks, -1, sizeof(bitblock), -1, nails, beta.data());

            // Memory Compare: GMP
            auto start = std::chrono::high_resolution_clock::now();
            int eq;
            for (unsigned int k = 0; k < samples; ++k) {
                eq = mpz_cmp(mpz_alpha, mpz_beta);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            gmp_compare += duration;

            // Clear necessary since mpz_export does not copy zero words (it assumes the destination memory has be cleared)
            alpha.clear();
            beta.clear();
            // Translation from mpz objects
            mpz_export(alpha.data(), NULL, -1, sizeof(bitblock), -1, nails, mpz_alpha);
            mpz_export(beta.data(), NULL, -1, sizeof(bitblock), -1, nails, mpz_beta);

            if (eq != 0) {
                std::cout << "GMP compare is incorrect." << std::endl;
                std::cout << "Alpha: " << alpha.to_string() << std::endl;
                std::cout << "Beta:  " << beta.to_string() << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Results:" << std::endl;
    std::cout << "  Custom Compare Average Runtime: " << (float)custom_compare / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  STD Compare Average Runtime: " << (float)std_compare / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  GMP Compare Average Runtime: " << (float)gmp_compare / (float)(trials * samples) << " ns" << std::endl;

    std::cout << "Benchmarking Bit Counting..." << std::endl;
    for (unsigned int i = 10; i < trials; ++i) {
        unsigned int expectation = 0;
        for (unsigned int j = 0; j < length; ++j) { if (j % 100 < i) { ++expectation; } }
        {
            // Set-up
            Bitmask alpha(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
            }

            // Bit Count: Custom
            auto start = std::chrono::high_resolution_clock::now();
            unsigned int cnt;
            for (unsigned int k = 0; k < samples; ++k) {
                cnt = alpha.count();
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            custom_count += duration;
            if (cnt != expectation) {
                std::cout << "Custom count is incorrect." << std::endl;
                exit(1);
            }
        }
        {
            // Set-up
            Bitmask alpha(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
            }
            // Translation to mpz objects
            mpz_t mpz_alpha;
            mpz_init2(mpz_alpha, length);
            mpz_import(mpz_alpha, blocks, -1, sizeof(bitblock), -1, nails, alpha.data());

            // Bit Count: GMP
            auto start = std::chrono::high_resolution_clock::now();
            unsigned int cnt;
            for (unsigned int k = 0; k < samples; ++k) {
                cnt = mpz_popcount(mpz_alpha); 
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            gmp_count += duration;

            // Clear necessary since mpz_export does not copy zero words (it assumes the destination memory has be cleared)
            alpha.clear();
            // Translation from mpz objects
            mpz_export(alpha.data(), NULL, -1, sizeof(bitblock), -1, nails, mpz_alpha);

            if (cnt != expectation) {
                std::cout << "GMP compare is incorrect." << std::endl;
                std::cout << "Alpha: " << alpha.to_string() << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Results:" << std::endl;
    std::cout << "  Custom Count Average Runtime: " << (float)custom_count / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  GMP Count Average Runtime: " << (float)gmp_count / (float)(trials * samples) << " ns" << std::endl;

    std::cout << "Benchmarking Logical AND..." << std::endl;
    for (unsigned int i = 10; i < trials; ++i) {
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, (j+i) % 100 < i);
            }

            // Logical AND: Custom
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                alpha.bit_and(beta);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            custom_and += duration;
        }
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, (j+i) % 100 < i);
            }
            // Translation to mpz objects
            mpz_t mpz_alpha;
            mpz_t mpz_beta;
            mpz_t mpz_gamma;
            mpz_init2(mpz_alpha, length);
            mpz_init2(mpz_beta, length);
            mpz_init2(mpz_gamma, length);
            mpz_import(mpz_alpha, blocks, -1, sizeof(bitblock), -1, nails, alpha.data());
            mpz_import(mpz_beta, blocks, -1, sizeof(bitblock), -1, nails, beta.data());

            // Logical AND: GMP
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                mpz_and(mpz_gamma, mpz_alpha, mpz_beta);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            gmp_and += duration;
        }
    }
    std::cout << "Results:" << std::endl;
    std::cout << "  Custom Logical AND Average Runtime: " << (float)custom_and / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  GMP Logical AND Average Runtime: " << (float)gmp_and / (float)(trials * samples) << " ns" << std::endl;

    std::cout << "Benchmarking Logical OR..." << std::endl;
    for (unsigned int i = 10; i < trials; ++i) {
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, (j+i) % 100 < i);
            }

            // Logical OR: Custom
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                alpha.bit_or(beta);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            custom_or += duration;
        }
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, (j+i) % 100 < i);
            }
            // Translation to mpz objects
            mpz_t mpz_alpha;
            mpz_t mpz_beta;
            mpz_t mpz_gamma;
            mpz_init2(mpz_alpha, length);
            mpz_init2(mpz_beta, length);
            mpz_init2(mpz_gamma, length);
            mpz_import(mpz_alpha, blocks, -1, sizeof(bitblock), -1, nails, alpha.data());
            mpz_import(mpz_beta, blocks, -1, sizeof(bitblock), -1, nails, beta.data());

            // Logical OR: GMP
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                mpz_ior(mpz_gamma, mpz_alpha, mpz_beta);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            gmp_or += duration;
        }
    }
    std::cout << "Results:" << std::endl;
    std::cout << "  Custom Logical OR Average Runtime: " << (float)custom_or / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  GMP Logical OR Average Runtime: " << (float)gmp_or / (float)(trials * samples) << " ns" << std::endl;

    std::cout << "Benchmarking Logical XOR..." << std::endl;
    for (unsigned int i = 10; i < trials; ++i) {
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, (j+i) % 100 < i);
            }

            // Logical XOR: Custom
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                alpha.bit_xor(beta);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            custom_xor += duration;
        }
        {
            // Set-up
            Bitmask alpha(length);
            Bitmask beta(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
                beta.set(j, (j+i) % 100 < i);
            }
            // Translation to mpz objects
            mpz_t mpz_alpha;
            mpz_t mpz_beta;
            mpz_t mpz_gamma;
            mpz_init2(mpz_alpha, length);
            mpz_init2(mpz_beta, length);
            mpz_init2(mpz_gamma, length);
            mpz_import(mpz_alpha, blocks, -1, sizeof(bitblock), -1, nails, alpha.data());
            mpz_import(mpz_beta, blocks, -1, sizeof(bitblock), -1, nails, beta.data());

            // Logical XOR: GMP
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                mpz_xor(mpz_gamma, mpz_alpha, mpz_beta);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            gmp_xor += duration;
        }
    }
    std::cout << "Results:" << std::endl;
    std::cout << "  Custom Logical XOR Average Runtime: " << (float)custom_xor / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  GMP Logical XOR Average Runtime: " << (float)gmp_xor / (float)(trials * samples) << " ns" << std::endl;

    std::cout << "Benchmarking Hash Function..." << std::endl;
    for (unsigned int i = 10; i < trials; ++i) {
        {
            // Set-up
            Bitmask alpha(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
            }

            // Hash Function: Custom
            auto start = std::chrono::high_resolution_clock::now();
            size_t code;
            for (unsigned int k = 0; k < samples; ++k) {
                code = alpha.hash(false);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            custom_hash += duration;
        }
        {
            // Set-up
            Bitmask alpha(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
            }

            // Hash Function: GMP
            auto start = std::chrono::high_resolution_clock::now();
            size_t code;
            for (unsigned int k = 0; k < samples; ++k) {
                code = alpha.hash(true);
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            gmp_hash += duration;
        }
    }
    std::cout << "Results:" << std::endl;
    std::cout << "  Full Scan Hash Function Average Runtime: " << (float)custom_hash / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  Bit Scan Hash Function Average Runtime: " << (float)gmp_hash / (float)(trials * samples) << " ns" << std::endl;

    std::cout << "Benchmarking Iteration..." << std::endl;
    for (unsigned int i = 10; i < trials; ++i) {
        {
            // Set-up
            Bitmask alpha(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
            }

            // Iteration: Custom
            unsigned int cnt = 0;
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                cnt = 0;
                for (unsigned q = 0; q < length; ++q) {
                    if (alpha.get(q)) { ++cnt; }
                }
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            custom_iterate += duration;
            if (cnt != alpha.count()) {
                std::cout << "Custom iteration is incorrect." << std::endl;
                std::cout << "Alpha: " << alpha.to_string() << std::endl;
                exit(1);
            }
        }
        {
            // Set-up
            Bitmask alpha(length);
            for (unsigned int j = 0; j < length; ++j) {
                alpha.set(j, j % 100 < i);
            }

            // Iteration: GMP
            unsigned int cnt;
            auto start = std::chrono::high_resolution_clock::now();
            for (unsigned int k = 0; k < samples; ++k) {
                cnt = 0;
                bool sign = alpha.get(0);
                unsigned int p = 0;
                unsigned int q = alpha.scan(p, !sign);
                unsigned int max = length;
                while (q <= max) {
                    if (sign) { cnt += q - p; }
                    if (q == max) { break; }
                    p = q;
                    sign = !sign;
                    q = alpha.scan(p, !sign);
                }
            }
            auto finish = std::chrono::high_resolution_clock::now();
            float duration = std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start).count();
            gmp_iterate += duration;
            if (cnt != alpha.count()) {
                std::cout << "GMP iteration is incorrect." << std::endl;
                std::cout << "Expected: " << alpha.count() << ", Got: " << cnt << std::endl;
                std::cout << "Alpha: " << alpha.to_string() << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Results:" << std::endl;
    std::cout << "  Full Scan Iteration Average Runtime: " << (float)custom_hash / (float)(trials * samples) << " ns" << std::endl;
    std::cout << "  Bit Scan Iteration Average Runtime: " << (float)gmp_hash / (float)(trials * samples) << " ns" << std::endl;
}
