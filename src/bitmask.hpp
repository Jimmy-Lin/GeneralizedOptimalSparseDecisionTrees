#ifndef BITMASK_H
#define BITMASK_H

#include <unordered_map>
#include <vector>
#include <boost/dynamic_bitset.hpp>
#include <tbb/scalable_allocator.h>

typedef boost::dynamic_bitset< unsigned long, tbb::scalable_allocator< unsigned long > > bitset;

class Bitmask {
public:
    // Convenience Constructor
    static Bitmask zeros(int const size);
    static Bitmask ones(int const size);
    static Bitmask fill(int const fill, int const size);

    Bitmask(void);
    Bitmask(boost::dynamic_bitset< unsigned long, tbb::scalable_allocator< unsigned long > > const & reference);

    boost::dynamic_bitset< unsigned long, tbb::scalable_allocator< unsigned long > > const & value(void) const;
    int const operator[](int index) const;
    void set(int const index, bool const value);
    int const count(void) const;
    int const size(void) const;
    size_t const hash(void) const;
    std::vector< int, tbb::scalable_allocator< int > > const & encoding(void) const;
//    std::vector< float > const dot(std::vector< float > const & other) const; //plswork

    bool const operator==(Bitmask const & other) const;
    bool const operator!=(Bitmask const & other) const;
    bool const operator<=(Bitmask const & other) const;
    bool const operator>=(Bitmask const & other) const;
    bool const operator<(Bitmask const & other) const;
    bool const operator>(Bitmask const & other) const;

    Bitmask const operator&(Bitmask const & other) const;
    Bitmask const operator^(Bitmask const & other) const;
    Bitmask const operator|(Bitmask const & other) const;
    Bitmask const operator-(Bitmask const & other) const;
    Bitmask const operator~(void) const;

    std::string const to_string(bool reverse = false) const;

    std::vector< unsigned long, tbb::scalable_allocator< unsigned long > > const dump(void) const;

    std::vector< bool > expand(void) const;


private:
    boost::dynamic_bitset< unsigned long, tbb::scalable_allocator< unsigned long > > reference;
    size_t hash_code;
    std::vector< int, tbb::scalable_allocator< int > > run_length_code;
};

// Hash implementation
namespace std {
    template <>
    struct hash< Bitmask > {
        std::size_t operator()(Bitmask const & bitmask) const {
            return bitmask.hash();
        }
    };
}

#endif
