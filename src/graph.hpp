#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <utility>
#include <unordered_map>
#include <unordered_set>

#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/cache_aligned_allocator.h>
#include "key.hpp"
#include "key_pair.hpp"
#include "task.hpp"

// Additional Hash Implementation for tbb::concurrent_hash_table
// These delegate to the already implemented hash functions and equality operators
struct GraphIndexHash {
    std::size_t operator()(Key const & key) const {
        return std::hash< Key >()(key);
    }
};

struct GraphVertexHashComparator {
    static size_t hash(Key const & key) {
        return std::hash< Key >()(key);
    }
    static bool equal(Key const & left, Key const & right) {
        return left == right;
    }
};

struct GraphEdgeHashComparator {
    static size_t hash(KeyPair const & pair) {
        return pair.hash();
        // return std::hash< KeyPair >()(pair);
    }
    static bool equal(KeyPair const & left, KeyPair const & right) {
        return left == right;
    }
};
 
typedef tbb::concurrent_unordered_set< Key, GraphIndexHash, std::equal_to< Key >, tbb::scalable_allocator< Key > > index_type;
typedef tbb::concurrent_hash_map< Key, index_type, GraphVertexHashComparator, tbb::cache_aligned_allocator< std::pair< Key, index_type > > > index_table;
typedef tbb::concurrent_hash_map< Key, Task, GraphVertexHashComparator, tbb::cache_aligned_allocator< std::pair< Key, Task > > > task_table;

class Graph {
public:
    int hits = 0;

    // Vertices
    task_table tasks;
    // (forward and backward indices)
    index_table backward_index;

    Graph(void);
    ~Graph(void);
};

#endif
