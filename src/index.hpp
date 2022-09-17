#ifndef INDEX_H
#define INDEX_H

#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
//#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/vector_proxy.hpp>
//#include <boost/numeric/ublas/vector_expression.hpp>
#include <tbb/scalable_allocator.h>
//#include <simdpp/simd.h>

#include "integrity_violation.hpp"
#include "bitmask.hpp"

// @note: vector type used for representing GPU floating point vector
//typedef boost::numeric::ublas::vector< float > blasvector;
// @note: vector type used for representing GPU mask
//typedef boost::numeric::ublas::vector< bitblock > blasmask;

// Container used to store prefix sums of vectors which help accelerate our calculations
// multiple vectors are stored in this container so that ranges don't need to be recomputed for each vector
class Index {
public:
    static void precompute(void);
    void benchmark(void) const;

    Index(void);
    // @param source: vector of floating points to sum over (efficiently)
    Index(std::vector< std::vector< float > > const & source);
    ~Index(void);

    // @param indicator: mask of bits indicating which elements are relevant to the vector sum
    // @returns the total of all elements associated to bits that were set to 1
    void sum(Bitmask const & indicator, float * accumulator) const;

    // @returns string representation of original floating points (used for inspection)
    std::string to_string(void) const;

private:
    // Copy of the original floating points
    std::vector< float > source;
    // precomputed representation of the floating point vector
    std::vector< std::vector< float > > prefixes;
    // Number of floating points represented in the source vector
    unsigned int size;
    unsigned int width;
    // Number of blocks expected in bitmask
    unsigned int num_blocks;

    // @param indicator: array of blocks of bits indicating which elements are relevant to the vector sum
    // @returns the total of all elements associated to bits that were set to 1
    // @note: This implementation uses look-up to precomputed values of the run-length-code for fast sums
    void block_sequential_sum(bitblock * blocks, float * accumulator) const;
    void block_sequential_sum(rangeblock block, unsigned int offset, float * accumulator) const;

    // @param indicator: array of blocks of bits indicating which elements are relevant to the vector sum
    // @returns the total of all elements associated to bits that were set to 1
    // @note: This implementation computes run-length-code for fast sums
    void bit_sequential_sum(Bitmask const &indicator, float *accumulator) const;

    // @param source: The original vector of floats used in computation
    // @modifies prefixes: writes the prefix sums into this vector
    void build_prefixes(std::vector< std::vector< float > > const & source, std::vector< std::vector< float > > & prefixes);
};
#endif
