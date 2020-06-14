#ifndef TASK_H
#define TASK_H

#include <vector>
#include <tuple>
#include <tbb/scalable_allocator.h>

#include "key.hpp"

typedef std::pair<float, float> bounding_pair;
typedef std::unordered_map<unsigned int, float, std::hash<unsigned int>, std::equal_to<unsigned int>, tbb::scalable_allocator< std::pair<const unsigned int, float> > > similarity_index_type;
typedef std::unordered_map<unsigned int, similarity_index_type, std::hash<unsigned int>, std::equal_to<unsigned int>, tbb::scalable_allocator<std::pair<const unsigned int, similarity_index_type>>> similarity_index_table_type;

class Task {
public:
    Task(void);
    Task(float const lowerbound, float const upperbound, float const support, float base_objective);
    Task(float const lowerbound, float const upperbound, float const support, float const base_objective, Bitmask const & sensitivity, similarity_index_table_type const & index);

    float const support(void) const;
    float const lowerbound(void) const;
    float const upperbound(void) const;
    float const potential(void) const;
    float const uncertainty(void) const;
    float const objective(void) const;
    float const base_objective(void) const;
    float const scope(void) const;
    void rescope(float scope_value);

    Bitmask const & sensitivity(void) const;
    
    bool const sensitive(int const index) const;
    void desensitize(int const index);

    float const priority(float const optimism) const;

    void inform(float const lowerbound, float const upperbound);

    bool const explored(void) const;
    bool const delegated(void) const;
    bool const cancelled(void) const;
    bool const resolved(void) const;

    void explore(void);
    void delegate(void);
    void cancel(void);
    void resolve(void);

    std::unordered_map<unsigned int, bounding_pair, std::hash<unsigned int>, std::equal_to<unsigned int>, tbb::scalable_allocator<std::pair<const unsigned int, bounding_pair>>> combined_bounds;
    similarity_index_table_type similarity_index;

private:
    float _support;
    float _lowerbound;
    float _upperbound;
    float _potential;
    float _base_objective;
    float _scope = 1.0;

    Bitmask _sensitivity;

    bool _explored = false;
    bool _delegated = false;
    bool _resolved = false;
    bool _cancelled = false;
};

#endif