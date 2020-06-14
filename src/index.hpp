#ifndef INDEX_H
#define INDEX_H

#define SIMDPP_ARCH_X86_SSE4_1
#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>
#include <simdpp/simd.h>

#include "bitmask.hpp"

typedef boost::numeric::ublas::vector< float > blasvector;
typedef boost::numeric::ublas::vector< unsigned int > blasmask;

class Index {
    public:
        static unsigned int rshifts[32];
        Index(void);
        Index(std::vector< float > const & source);
        ~Index(void);
        float sum(void) const;
        float sum(Bitmask const & indicator) const;
        Index zeroOut(Bitmask const & other) const;

    private:
        blasvector source;
        std::vector< float > nativeSource;
        std::vector< float > data;
        unsigned int size;

        void build(std::vector< float > const & source, std::vector< float > & tree, unsigned int target, unsigned int left, unsigned int right);
        void buildDP(std::vector< float > const & source, std::vector< float > & tracker);
        float sum(unsigned int query_left, unsigned int query_right) const;
        float sum(unsigned int target, unsigned int left, unsigned int right, unsigned int query_left, unsigned int query_right) const;
        float sumDP(unsigned int query_left, unsigned int query_right) const;

};

#endif