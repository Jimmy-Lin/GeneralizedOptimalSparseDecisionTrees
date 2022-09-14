#ifndef BITMASK_H
#define BITMASK_H

#include <chrono>
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <gmp.h>
//#include <boost/dynamic_bitset.hpp>
//#include <simdpp/simd.h>
#include <tbb/scalable_allocator.h>

#include "integrity_violation.hpp"

typedef mp_limb_t bitblock; // Type used to store binary bits
typedef unsigned short rangeblock; // Type used to chunk the binary bits into  precomputable sequences
typedef char codeblock; // Type used to store run-length codes for each precomputable sequence

//typedef boost::dynamic_bitset< unsigned long long > dynamic_bitset;

// This declaration acts as both a function module and a container class
// The static class methods implements a function module providing operations on arrays of type bitblock, which can be allocated on the stack
// The non-static class methods implementes a heap-allocated equivalent, which supports the same methods
// @note: Many of the binary operations assume that operands have the same length
class Bitmask {
public:
    static const bitblock ranges_per_code; // Number of ranges that can be encoded using a single codeblock
    static const bitblock bits_per_range; // Number of bits in a single rangeblock
    static const bitblock bits_per_block; // Number of bits in a single bitblock

    // @param blocks: the blocks containing bits
    // @param size: the number of bits which are represented in blocks
    // @modifies blocks will be set to have all bits be 1
    static void ones(bitblock * blocks, unsigned int size);

    // @param blocks: the blocks containing bits
    // @param size: the number of bits which are represented in blocks
    // @modifies blocks will be set to have all bits be 0
    static void zeros(bitblock * blocks, unsigned int size);

    // @param blocks: the blocks containing bits to copy
    // @param other_blocks: the blocks which will be overwritten by the copy
    // @param size: the number of bits which are represented in blocks
    // @modifies other_blocks will be overwritten with the contents of blocks
    static void copy(bitblock * blocks, bitblock * other_blocks, unsigned int size);

    // @param size: the number of bits to be represented
    // @param number_of_blocks: pointer to store the number of blocks required to represent these bits
    // @param block_offset: pointer to store the number of bits actually used in the last block
    // @modifies number_of_bits: initialized with the number of blocks required to represent these bits
    // @modifies block_offset: initialized with the number of bits actually used in the last block
    static void block_layout(unsigned int size, unsigned int * number_of_blocks, unsigned int * block_offset);

    // @param blocks: the blocks to sanitize / normalize
    // @param number of blocks: the number of blocks present in blocks
    // @param offset: the number of bits present in the final block
    // @modifies blocks: the final block will have all unsused bits cleared
    static void clean(bitblock * blocks, unsigned int number_of_blocks, unsigned int offset);

    // @param blocks: the blocks from which to count the bits which are set to 1
    // @param size: the number of bits which are represented in blocks
    // @returns the number of represented bits which are set to 1
    static unsigned int count(bitblock * blocks, unsigned int size);

    // @param blocks: the blocks from which to compute the number of contiguous ranges of bits which are set to 1
    // @param size: the number of bits which are represented in blocks
    // @returns the estimated number of contiguous ranges of bits which are set to 1
    // @note the this was originally an approximate method. I later altered this to an exact method :D
    static unsigned int words(bitblock * blocks, unsigned int size);

    // @param blocks: blocks which act as a vector of bits for a bit-wise logical and
    // @param other_blocks: blocks which act as a vector of bits for a bit-wise logical and
    // @param size: the number of bits which are represented in blocks and other_blocks
    // @param flip: whether or not to interpret the bits of blocks as flipped before applying the bit-wise operation
    // @modifies other_blocks: will be overwritten with the result
    static void bit_and(bitblock * blocks, bitblock * other_blocks, unsigned int size, bool flip = false);
    
    // @param blocks: blocks which act as a vector of bits for a bit-wise logical or
    // @param other_blocks: blocks which act as a vector of bits for a bit-wise logical or
    // @param size: the number of bits which are represented in blocks and other_blocks
    // @param flip: whether or not to interpret the bits of blocks as flipped before applying the bit-wise operation
    // @modifies other_blocks: will be overwritten with the result
    static void bit_or(bitblock * blocks, bitblock * other_blocks, unsigned int size, bool flip = false);

    // @param blocks: blocks which act as a vector of bits for a bit-wise logical xor
    // @param other_blocks: blocks which act as a vector of bits for a bit-wise logical xor
    // @param size: the number of bits which are represented in blocks and other_blocks
    // @param flip: whether or not to interpret the bits of blocks as flipped before applying the bit-wise operation
    // @modifies other_blocks: will be overwritten with the result
    static void bit_xor(bitblock * blocks, bitblock * other_blocks, unsigned int size, bool flip = false);

    // @param blocks: blocks which act as a vector of bits for comparison
    // @param other_blocks: blocks which act as a vector of bits for comparison
    // @param size: the number of bits which are represented in blocks and other_blocks
    // @param flip: whether or not to interpret the bits of blocks as flipped before applying the bit-wise operation
    // @returns whether blocks is bit-wise equal to other_blocks
    static bool equals(bitblock * const blocks, bitblock * const other_blocks, unsigned int size, bool flip = false);

    // @param left: blocks which act as a vector of bits for comparison
    // @param right: blocks which act as a vector of bits for comparison
    // @param size: the number of bits which are represented in blocks and other_blocks
    // @returns 0 if left == right, -1 if left < right, 1 if left > right
    // @note this comparison uses bit 0 as the least significant bit (little endian)
    static int compare(bitblock * const left, bitblock * const right, unsigned int size);

    // @param left: blocks which will be compared with right
    // @param right: blocks which will be compared with left
    // @param size: the number of blocks to use during comparison
    // @returns whether left is less (or greater) than right
    // @note this comparison uses bit 0 as the least significant bit (little endian)
    static bool less_than(bitblock * const left, bitblock * const right, unsigned int size);
    static bool greater_than(bitblock * const left, bitblock * const right, unsigned int size);

    // @param blocks: blocks which contain bits to be hashed
    // @param size: the number of bits which are represented in blocks
    // @returns a hash value of blocks
    static size_t hash(bitblock * blocks, unsigned int size);

    // @param blocks: blocks which contain bits to be read
    // @param size: the number of bits which are represented in blocks
    // @param index: index of the bit to be read
    // @returns the bit at the provided index represented as either 1 or 0
    static unsigned int get(bitblock * blocks, unsigned int size, unsigned int index);

    // @param blocks: blocks which contain bits to be written
    // @param size: the number of bits which are represented in blocks
    // @param index: index of the bit to be written
    // @param value: value to be written to at indexed position
    // @modifies blocks: blocks will be modified at the specified index with the specified value
    static void set(bitblock * blocks, unsigned int size, unsigned int index, bool value = 1);

    // @param blocks: blocks of bits to scan over
    // @param size: the number of bits which are represented in blocks
    // @param start: the starting index (inclusive) to scan for a bit that matches 'value'
    // @returns: the index of the first bit that matches value
    // @note: scan will search from 'start' towards more significant bits
    // @note: rscan will search from 'start' towards less significant bits
    static int scan(bitblock * blocks, int size, int start, bool value = 1);
    static int rscan(bitblock * blocks, int size, int start, bool value = 1);

    // @param blocks: blocks which will be converted to string format
    // @param size: the number of bits expected to convert
    // @param reverse: whether to reverse the printed bit order
    // @returns a string containing the bits in order of least significant to most significant
    static std::string to_string(bitblock * blocks, unsigned int size, bool flip = false);

    static bool integrity_check; // Flag that indicates whether null pointer checks are performed on object instances
    static bool precomputed; // Flag that indicates whether look-up tables are completed
    static std::vector< std::vector< codeblock > > ranges; // Precomputed run-length codes in rangeblock-sized sequences
    static std::vector< size_t > hashes; // Precomputed hashes in rangeblock-sized sequences
    static std::vector< char > counts; // Precomputed population counts in rangeblock-sized sequences

    static void precompute(void); // Perform the one-time precomputation for rangeblock-sized sequences
    static void benchmark(unsigned int size); // Run a benchmark for a bitmask of the given size

    Bitmask(void);
    // Construction from a single fill-value and size
    Bitmask(unsigned int size, bool fill = false, bitblock * local_buffer=NULL, unsigned char depth_budget = 0);
    // Construction by copying from a stack-based bitblock array
    Bitmask(bitblock * blocks, unsigned int block_count, bitblock * local_buffer=NULL, unsigned char depth_budget = 0);
    // Construction by copying from a dynamic_bitset
    //Bitmask(dynamic_bitset const & source, bitblock * local_buffer=NULL, unsigned char depth_budget = 0);
    // Construction by copying from another instance
    Bitmask(Bitmask const & source, bitblock * local_buffer=NULL);

    ~Bitmask(void);

    // Initialize attributes and allocate memory (if needed) for this instance
    void initialize(unsigned int size, bitblock * local_buffer=NULL);

    // @param dest_blocks: an array of bitblocks to store bits on
    // @modifies dest_blocks: the contents of dest_blocks will be overwritten with a copy of the bits stored in this bitmask
    void copy_to(bitblock * dest_blocks) const;

    // @param dest_blocks: an array of bitblocks to copy bits from
    // @modifies content: the instance member content will be overwritten with _size blocks of src_blocks
    void copy_from(bitblock * src_blocks);
    // Aliases to copy_from
    Bitmask & operator=(Bitmask const & other);

    // @returns a pointer to the blocks of this instance
    bitblock * data(void) const;

    // @returns the number of bits represented by this instance
    unsigned int size(void) const;

    // @returns the number of bits representable by this instance
    unsigned int capacity(void) const;

    // @returns the number of bits set to 1
    unsigned int count(void) const;

    // @returns estimates the number of contiguous ranges of 1's
    unsigned int words(void) const;

    // @param start: the bit index to start scanning from
    // @param value the value of the bit to scan for
    // @returns the index of the first bit that matches value, starting at (inclusive) the index 'start'.
    // @note scan returns size if scan doesn't find any matches, -1 if rscan doesn't find any matches
    // @note the scan implementation searches from start to most-significant bit
    // @note the rscan implementation searches from start to least-significant bit
    int scan(int start, bool value) const;
    int rscan(int start, bool value) const;

    // @param value the value of the bit to scan for
    // @param begin: the bit index to start scanning from
    // @modifies begin: the index of the first bit in the scan that matches value
    // @modifies end: the index of the first bit that doesn't match the value following the bit matched at 'begin'
    // @return true if a range is found
    // @note these implementations are for finding contiguous ranges of bits matching a specific value
    bool scan_range(bool value, int & begin, int & end) const;
    bool rscan_range(bool value, int & begin, int & end) const;

    // @returns true if and only if all bits are set to 0
    bool empty(void) const;

    // @returns true if and only if all bits are set to 1
    bool full(void) const;

    // @param blocks: the second operand vector of blocks of bits
    // @param flip: whether or not to treat the bits of this instance as flipped before applying the operation
    // @modifies blocks: blocks will be overwritten with the resulting bits
    void bit_and(bitblock * blocks, bool flip = false) const;
    void bit_xor(bitblock * blocks, bool flip = false) const;
    void bit_or(bitblock * blocks, bool flip = false) const;
    void bit_and(Bitmask const & other, bool flip = false) const;
    void bit_xor(Bitmask const & other, bool flip = false) const;
    void bit_or(Bitmask const & other, bool flip = false) const;

    // @modifes: sets every bit to zero
    void clear(void);

    // @modifes: sets every bit to one
    void fill(void);

    // @param blocks: an array of bitblocks containing bits to be compared
    // @returns whether the bits match up to the size of this instance
    bool operator==(bitblock * blocks) const;

    // @param other: a bitmask containing an array of bitblocks containing bits to be compared
    // @returns whether the bits match up to the size of this instance
    bool operator==(Bitmask const & other) const;

    // @param other: another bitmask instance for comparison
    // @returns whether the other bitmask is considered lesser
    // @note the 0th bit is considered most significant when making this comparison
    bool operator<(Bitmask const & other) const;

    // @param other: another bitmask instance for comparison
    // @returns whether the other bitmask is considered lesser
    // @note the 0th bit is considered most significant when making this comparison
    bool operator>(Bitmask const & other) const;

    // Additional derived relational operators
    bool operator!=(Bitmask const & other) const;
    bool operator<=(Bitmask const & other) const;
    bool operator>=(Bitmask const & other) const;

    // @returns the hash value of this bitmask
    size_t hash(bool bitwise = true) const;

    // @param index: an index specifying which bit to read
    // @returns: the accessed bit in the form of an integer
    unsigned int get(unsigned int index) const;
    // @note: aliases to get
    unsigned int operator[](unsigned int index) const;

    // @param index: an index specifying which bit to modify
    // @param value: a boolean value specifying whether to set the bit to 1 or 0
    void set(unsigned int index, bool value = true);

    // @returns: the depth budget for the subproblem this bitmask represents (assuming the bitmask represents a subproblem)
    unsigned char get_depth_budget() const;

    // @param: depth_budget: the depth budget for the subproblem this bitmask represents. 
    void set_depth_budget(unsigned char depth_budget);

    // @param reverse: reverses the sequence so that the 0th bit is on the right, as opposed to on the left (little endian)
    // @returns a string representing the bit sequence with each bit represented as the character '1' or '0'
    // @note default reverses the sequence so that bit-0 is the left most in the string
    std::string to_string(bool reverse = false) const;

    // @returns true if the content of the object passes the null pointer check
    bool valid(void) const;
    // @throws an exception if this object contains a null pointer
    void validate(void) const;

    // @requires the new_size must be less that _max_blocks * bits_per_block. (i.e. the maximum capacity set during construction)
    // @modifies _size: modified to the new size
    // @modifies _used_blocks: modified to the new number of used blocks
    // @modifies _offset: modified to the new number of bits used in the last used block
    // @note: resizing does not affect memory usage. allocated memory remain constant until this instance is deconstructed
    void resize(unsigned int _new_size); // The number of bits actually being used (excludes leading zeros in final block)

private:
    unsigned char depth_budget; // If Bitmask represents subproblem - maximum allowable depth for the solution 
                                // 1 means no further splits allowed - a (sub)tree with only one node is depth 1.
                                // 0 means there is no constraint on allowable depth

    static tbb::scalable_allocator<bitblock> allocator; // Allocator used to managing memory
    bitblock * content = NULL; // A pointer the blocks containing the bits
    unsigned int _size = 0; // The number of bits actually being used (excludes leading zeros in final block)
    unsigned int _offset = 0; // The number of bits used in the last used block by this object instance
    unsigned int _used_blocks = 0; // The number of blocks currently used by this object instance
    unsigned int _max_blocks = 0; // The number of blocks occupied by this object instance

    // If shallow is true, then the destructor is not responsible for deallocating the memory pointed to by content
    bool shallow = false;
};

// Overrides for STD containers
namespace std {
    template <>
    struct hash< Bitmask > {
        std::size_t operator()(Bitmask const & bitmask) const { return bitmask.hash(); }
    };

    template <>
    struct less< Bitmask > {
        bool operator()(Bitmask const & left, Bitmask const & right) const { return left < right; }
    };

    template <>
    struct equal_to< Bitmask > {
        bool operator()(Bitmask const & left, Bitmask const & right) const { return left == right; }
    };
}

#endif
