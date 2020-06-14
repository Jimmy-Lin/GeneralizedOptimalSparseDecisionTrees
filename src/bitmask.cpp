#include "bitmask.hpp"

Bitmask Bitmask::zeros(int const size) {
    return fill(0, size);
}

Bitmask Bitmask::ones(int const size) {
    return fill(1, size);
}

Bitmask Bitmask::fill(int const fill, int const size) {
    // boost::dynamic_bitset<> value(size);
    // if (fill == 1) {
    //     value.flip();
    // }
    bitset value;
    value.resize(size, fill);
    return Bitmask(value);
}

Bitmask::Bitmask(void) {}

Bitmask::Bitmask(bitset const & reference) : reference(reference) {
    // Perform Run-length encoding to compress the data
    int prior = 0;
    int length = 0;
    int end = reference.size();
    std::vector< int, tbb::scalable_allocator< int > > & run_length_code = this -> run_length_code;
    for (int i = 0; i < end; ++i) {
        int bit = reference[i];
        if (bit == prior) {
            ++length;
        } else {
            run_length_code.emplace_back(length);
            prior = bit;
            length = 0;
        }
    }
    // Further compute a hash based on the compressed code
    size_t seed = 0;

    // Hash the sums
    seed ^=  (reference.count()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    seed ^=  (run_length_code.size()) + 0x9e3779b9 + (seed<<6) + (seed>>2);

    if (true || run_length_code.size() <=  reference.num_blocks()) {
        // Hash the compressed encoding
        for (auto iterator = run_length_code.begin(); iterator != run_length_code.end(); ++iterator) {
            seed ^=  (* iterator) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
    } else {
        // Hash the literal content
        std::vector< unsigned long, tbb::scalable_allocator< unsigned long > > blocks(this -> reference.num_blocks());
        boost::to_block_range(this -> reference, blocks.begin());
        for (auto iterator = blocks.rbegin(); iterator != blocks.rend(); ++iterator) {
            seed ^=  (* iterator) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
    }
    this -> hash_code = seed;
}

bitset const & Bitmask::value(void) const {
    return this -> reference;
}

int const Bitmask::operator[](int index) const {
    return this -> reference[index];
}

void Bitmask::set(int const index, bool const value) {
    this -> reference[index] = value;
    return;
}


int const Bitmask::count(void) const {
    return this -> reference.count();
}

int const Bitmask::size(void) const {
    return this -> reference.size();
}

size_t const Bitmask::hash(void) const {
    return this -> hash_code;
}

std::vector< int, tbb::scalable_allocator< int > > const & Bitmask::encoding(void) const {
    return this -> run_length_code;
}

std::vector< unsigned long, tbb::scalable_allocator< unsigned long > > const Bitmask::dump(void) const {
    std::vector< unsigned long, tbb::scalable_allocator< unsigned long > > blocks(this -> reference.num_blocks());
    boost::to_block_range(this -> reference, blocks.begin());
    return blocks;
}

//std::vector< float > const Bitmask::dot(std::vector< float > const & other) {
//    std::vector< float > ret;
//    for (unsigned int i = 0; i < other.size(); i++) {
//        if (this -> reference[i] == 0) {
//            ret.emplace_back(0);
//        } else {
//            ret.emplace_back(other[i]);
//        }
//    }
//    return ret;
//}

bool const Bitmask::operator==(Bitmask const & other) const {
    // return (this -> reference.count() == other.reference.count()) && (this -> reference == other.reference);

    if (this -> hash() != other.hash() || this -> run_length_code.size() != other.run_length_code.size()) {
        return false;
    } else {
        if (this -> run_length_code.size() <= this -> reference.num_blocks()) {
            return this -> run_length_code == other.run_length_code;
        } else {
            return (this -> reference.count() == other.reference.count()) && (this -> reference == other.reference);
        }
    }
}

bool const Bitmask::operator!=(Bitmask const & other) const {
    return (this -> reference.count() != other.reference.count()) || (this -> reference != other.reference);
}

bool const Bitmask::operator<=(Bitmask const & other) const {
    return this -> reference <= other.reference;
}
bool const Bitmask::operator>=(Bitmask const & other) const {
    return this -> reference >= other.reference;
}
bool const Bitmask::operator<(Bitmask const & other) const {
    return this -> reference < other.reference;
}
bool const Bitmask::operator>(Bitmask const & other) const {
    return this -> reference > other.reference;
}

Bitmask const Bitmask::operator&(Bitmask const & other) const {
    return Bitmask(this -> reference & other.reference);
}

Bitmask const Bitmask::operator^(Bitmask const & other) const {
    return Bitmask(this -> reference ^ other.reference);
}

Bitmask const Bitmask::operator|(Bitmask const & other) const {
    return Bitmask(this -> reference | other.reference);
}

Bitmask const Bitmask::operator-(Bitmask const & other) const {
    return Bitmask(this -> reference - other.reference);
}

Bitmask const Bitmask::operator~(void) const {
    return Bitmask(~this -> reference);
}

std::string const Bitmask::to_string(bool reverse) const {
    std::string bitstring;
    boost::to_string(this -> reference, bitstring);
    if (reverse) { std::reverse(bitstring.begin(),bitstring.end()); }
    return bitstring;
}
