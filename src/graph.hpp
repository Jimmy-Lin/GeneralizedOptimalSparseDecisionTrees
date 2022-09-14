#ifndef GRAPH_H
#define GRAPH_H

#include "bitmask.hpp"
#include "task.hpp"
#include "tile.hpp"

#include <iostream>
#include <utility>
#include <vector>
#include <tuple>

#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/scalable_allocator.h>

//class Graph;

typedef Tile key_type;
typedef Task value_type;
typedef std::vector<int> translation_type;

// Additional Hash Implementation for tbb::concurrent_hash_table
// These delegate to the already implemented hash functions and equality operators
class GraphVertexHashComparator {
public:
    static size_t hash(key_type const & key) {
        return key.hash();
    }
    static bool equal(key_type const & left, key_type const & right) {
        return left == right;
    }
};

// class GraphTranslationHashComparator {
// public:
//     static size_t hash(std::pair<key_type, key_type> const & key) {
//         size_t seed = key.first.hash();
//         seed ^= key.second.hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         return seed;
//     }
//     static bool equal(std::pair<key_type, key_type> const & left, std::pair<key_type, key_type> const & right) {
//         return left == right;
//     }
// };

class GraphChildHashComparator {
public:
    static size_t hash(std::pair<key_type, int> const & key) {
        size_t seed = key.second;
        seed ^= key.first.hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
    static bool equal(std::pair<key_type, int> const & left, std::pair<key_type, int> const & right) {
        return left == right;
    }
};

typedef tbb::concurrent_hash_map< // Table for storing forward edges
    std::pair<key_type, int>, key_type, GraphChildHashComparator, 
    tbb::scalable_allocator<std::pair<std::pair<key_type, int> const, key_type>>> child_table;

typedef tbb::concurrent_hash_map< // Table for storing tile-orderings
    std::pair<key_type, int>, translation_type, GraphChildHashComparator, 
    tbb::scalable_allocator<std::pair<std::pair<key_type, int> const, translation_type>>> translation_table;

typedef tbb::concurrent_hash_map< // Table for storing vertices
    key_type, value_type, GraphVertexHashComparator, 
    tbb::scalable_allocator<std::pair<key_type const, value_type>>> vertex_table;

typedef tbb::concurrent_unordered_map< // Set of parents for a single vertex
    key_type, std::pair<Bitmask, float>, std::hash<Tile>, std::equal_to<Tile>, 
    tbb::scalable_allocator<std::pair<key_type const, std::pair<Bitmask, float>>>> adjacency_set; 

typedef tbb::concurrent_hash_map< // Table of all adjacency sets
    key_type, adjacency_set, GraphVertexHashComparator,
    tbb::scalable_allocator<std::pair<key_type const, adjacency_set>>> adjacency_table;

typedef tbb::concurrent_vector<std::tuple<unsigned int, float, float>, tbb::scalable_allocator<std::tuple<unsigned int, float, float>>> bound_list; // List of split-bounds for a single vertex

typedef tbb::concurrent_hash_map< // Table of all bound lists
    key_type, bound_list, GraphVertexHashComparator, 
    tbb::scalable_allocator<std::pair<key_type const, bound_list>>> bound_table;

typedef vertex_table::const_accessor const_vertex_accessor;
typedef vertex_table::accessor vertex_accessor;

typedef translation_table::const_accessor const_translation_accessor;
typedef translation_table::accessor translation_accessor;

typedef child_table::const_accessor const_child_accessor;
typedef child_table::accessor child_accessor;

typedef adjacency_table::const_accessor const_adjacency_accessor;
typedef adjacency_table::accessor adjacency_accessor;

typedef adjacency_set::const_iterator const_adjacency_iterator;
typedef adjacency_set::iterator adjacency_iterator;

// typedef adjacency_set::const_indicator const_adjacency_indicator;
// typedef adjacency_set::indicator adjacency_indicator;

typedef bound_table::const_accessor const_bound_accessor;
typedef bound_table::accessor bound_accessor;

typedef bound_list::const_iterator const_bound_iterator;
typedef bound_list::iterator bound_iterator;

// Container for storing the dependency graph
// The vertices of his graph act as a memoization table of subproblems
// Entries in the table are not necessarily complete, some are still running, paused, or cancelled.
class Graph {
public:
    translation_table translations;
    child_table children;
    vertex_table vertices;
    adjacency_table edges;
    bound_table bounds;

    Graph(void);
    ~Graph(void);

    // bool exists(key_type const & key) const;
    
    // bool insert(key_type const & key, value_type const & value);
    // bool insert(std::pair< key_type, value_type > const & pair);
    // bool connect(key_type const & parent, key_type const & child, float scope);

    // bool find(const_vertex_accessor & accessor, key_type const & key) const;
    // bool find(vertex_accessor & accessor, key_type const & key) const;

    // bool find(const_adjacency_accessor & accessor, key_type const & key, bool forward = true) const;
    // bool find(adjacency_accessor & accessor, key_type const & key, bool forward = true) const;

    // bool find_or_create(const_vertex_accessor & accessor, key_type const & key,
    // Bitmask & buffer_1, Bitmask & buffer_2, Bitmask & buffer_3,
    // Task const & task, unsigned int index, bool condition);

    // bool find_or_create(vertex_accessor & accessor, key_type const & key,
    // Bitmask & buffer_1, Bitmask & buffer_2, Bitmask & buffer_3,
    // Task const & task, unsigned int index, bool condition);

    bool erase(key_type const & key, bool disconnect = true);
    bool disconnect(key_type const & arent, key_type const & child);
    void clear(void);

    unsigned int size(void) const;
};

#endif
