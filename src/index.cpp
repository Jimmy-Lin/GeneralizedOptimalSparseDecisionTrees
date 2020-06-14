#include "index.hpp"

unsigned int Index::rshifts[32] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};

// The basic implementation is a segment tree
// However, there might be cases later on where it's faster to perform this using a BLAS library
Index::Index(void) {}

Index::Index(std::vector< float > const & source) : nativeSource(source) {
    this -> source.resize(source.size());
    std::copy(source.begin(), source.end(), this -> source.begin());
    this -> size = source.size();
    this -> data.resize(4 * source.size());
//    build(source, this -> data, 1, 0, source.size() - 1);
    buildDP(source, this -> data); // switch above line with this one to use segment tree
}

Index::~Index(void) {}

void Index::build(std::vector< float > const & source, std::vector< float > & tree, unsigned int target, unsigned int left, unsigned int right) {
    if (left == right) {
        tree[target] = source[left];
    } else {
        unsigned int split = (left + right) / 2; // Intentional Floor Division
        build(source, tree, target * 2, left, split); // Build Left Bisection
        build(source, tree, target * 2 + 1, split + 1, right); // Build Right Bisection
        tree[target] = tree[target * 2] + tree[target * 2 + 1]; // Compute the sums based on the built bisections
    }
}

// build dp table --> called tracker instead of tree
void Index::buildDP(std::vector< float > const & source, std::vector< float > & tracker) {
    tracker[0] = 0;
    for (int i = 0; i < source.size(); i++) {
        tracker[i+1] = source[i] + tracker[i];
    }
}

float Index::sum(void) const {
    return sum(0, size - 1);
}

float Index::sum(unsigned int query_left, unsigned int query_right) const {
//    return sum(1, 0, size - 1, query_left, query_right);
    return sumDP(query_left, query_right); // switch above line with this one to use segment tree
}

float Index::sum(unsigned int target, unsigned int left, unsigned int right, unsigned int query_left, unsigned int query_right) const {
    if (query_left > query_right) { return 0; }
    if (left == query_left && right == query_right) { return this -> data[target]; }
    unsigned int split = (left + right ) / 2; // Intentional Floor Division
    return sum(target * 2, left, split, query_left, std::min(query_right, split)) 
        + sum(target * 2 + 1, split + 1, right, std::max(query_left, split + 1), query_right);
}

float Index::sumDP(unsigned int query_left, unsigned int query_right) const {
    return (this -> data[query_right+1]) - (this -> data[query_left]);
}

// This function detects contiguous segments in the bitmask and performs a range query over the segment tree
// for each segment.
float Index::sum(Bitmask const & indicator) const {
    float total = 0.0;
    // Perform Run-length encoding to compress the data
    unsigned int prior = 0;
    unsigned int length = 1;
    unsigned int i;
    for (i = 0; i < size; ++i) {
        if (indicator[i] == prior) { ++length; } else {
            // The end of a contiguous selected segment 
//            if (prior == 1) { total += sum(i - length, i - 1); }
            if (prior == 1) { total += sumDP(i - length, i - 1); } // switch above line with this one to use segment tree
            prior = indicator[i];
            length = 1;
        }
    }
//    if (prior == 1) { total += sum(i - length, i - 1); } // Don't forget the final segment
    if (prior == 1) { total += sumDP(i - length, i - 1); } // Don't forget the final segment, switch above line with this one to use segment tree
    return total;
}

Index Index::zeroOut(Bitmask const & other) const {
    std::vector< float > temp = this -> data;
    for (unsigned int i = 0; i < other.size(); i++) {
        if (other[i] == 0) {
            temp[i] = 0;
        }
    }
    return Index(temp);
}

// float Index::sum(Bitmask const & indicator) const {
//     // Hash the literal content
//     std::vector< unsigned long, tbb::scalable_allocator< unsigned long > > blocks(indicator.value().num_blocks());
//     boost::to_block_range(indicator.value(), blocks.begin());
//     // Later convert this to stack memory
//     std::vector< unsigned int, tbb::scalable_allocator< unsigned int > > expansion(indicator.size());
//     // initialize in constructor
//     simdpp::uint32<32> * shifts = reinterpret_cast< simdpp::uint32<32> * >(Index::rshifts);
//     unsigned int block_index = 0;
//     for (auto block : blocks) {
//         {
//             simdpp::uint32<32> broadcast = simdpp::load_splat(&block);
//             simdpp::uint32<32> flags = simdpp::bit_and(simdpp::shift_r(broadcast, * shifts), 0x01);
//             simdpp::store( &(expansion[block_index * sizeof(unsigned long)]), flags);
//         }
//         block = block >> 32;
//         {
//             simdpp::uint32<32> broadcast = simdpp::load_splat(&block);
//             simdpp::uint32<32> flags = simdpp::bit_and(simdpp::shift_r(broadcast, * shifts), 0x01);
//             simdpp::store( &(expansion[32 + block_index * sizeof(unsigned long)]), flags);
//         }`
//         ++block_index;
//     }
//     blasmask mask(expansion.size());
//     std::copy(expansion.begin(), expansion.end(), mask.begin());
//     float result = boost::numeric::ublas::inner_prod(mask, this -> source);
//     return result;
// }

// float Index::sum(Bitmask const & indicator) const {
//     // Hash the literal content
//     std::vector< unsigned long, tbb::scalable_allocator< unsigned long > > blocks(indicator.value().num_blocks());
//     boost::to_block_range(indicator.value(), blocks.begin());


//     simdpp::mask_int64<size>* mblocks = reinterpret_cast<decltype(mblocks)>(blocks.data());

//     simdpp::uint64<size> zeros = simdpp::make_uint(0);

//     simdpp::float32<size>* vsources = reinterpret_cast<decltype(vsources)>(this->nativeSource);

//     auto rv = simdpp::blend(*vsources, zeros, *mblocks).eval();

//     return simdpp::reduce_add(rv);
// }
