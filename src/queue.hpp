#ifndef QUEUE_H
#define QUEUE_H

#include <iostream>
#include <tuple>
#include <unordered_set>

#include <tbb/concurrent_priority_queue.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/scalable_allocator.h>
#include <tbb/cache_aligned_allocator.h>
#include "key.hpp"

typedef std::tuple< Key, float, float, float > queue_element;

class PriorityKeyComparator {
public:
    // Note that the tbb::concurrent_priority_queue is implemented to pop the item with the highest priority value
    bool operator()(queue_element const & left, queue_element const & right) {
        // return left.indicator().count() > right.indicator().count();
        if (std::get< 1 >(left) != std::get< 1 >(right)) {
            return std::get< 1 >(left) < std::get< 1 >(right);
        } else if (std::get< 2 >(left) != std::get< 2 >(right)) {
            return std::get< 2 >(left) < std::get< 2 >(right);
        } else if (std::get< 3 >(left) != std::get< 3 >(right)) {
            return std::get< 3 >(left) < std::get< 3 >(right);
        } else {
            return true;
        }
    }
};

struct MembershipKeyHashCompare {
    static size_t hash(Key const & key) {
        return std::hash< Key >()(key);
    }
    static bool equal(Key const & left, Key const & right) {
        return left == right && left.feature_index() == right.feature_index();
    }
};

typedef tbb::concurrent_priority_queue< queue_element, PriorityKeyComparator, tbb::cache_aligned_allocator< queue_element > > queue_type;
typedef std::tuple< std::vector< float >, int > identity_type;

class Queue {
    tbb::concurrent_vector< identity_type, tbb::cache_aligned_allocator< identity_type > > identities;
    std::vector< queue_type, tbb::cache_aligned_allocator< queue_type > > queues;
    tbb::concurrent_hash_map< Key, bool, MembershipKeyHashCompare, tbb::cache_aligned_allocator< std::pair< Key, bool > > > membership;

    int _width;
    int _height;
    float _equity;

    float difference(int const index, Bitmask const & instance) const;
    float saturation(int const index) const;
    void assimilate(int const index, Bitmask const & instance);
public:
    Queue(void);
    ~Queue(void);
    void initialize(int const width, int const height, float const equity);
    void push(Key const & key, float const primary_priority = 0, float const secondary_priority = 0, float const tertiary_priority = 0);
    bool const empty(void) const;
    int const size(void) const;
    int const local_size(int const index) const;
    bool pop(queue_element & item, int const index);
    int const width(void) const;
    int const height(void) const;
};

#endif