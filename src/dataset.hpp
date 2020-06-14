#ifndef DATASET_H
#define DATASET_H

#define CL_SILENCE_DEPRECATION
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <sstream>
#include <math.h>
#include <map>
#include <vector>
#include <tuple>
#include <assert.h>
#include <boost/dynamic_bitset.hpp>
#include <tbb/concurrent_hash_map.h>
#include <tbb/scalable_allocator.h>

#ifdef INCLUDE_OPENCL
#include <opencl/cl.hpp>
#endif

#include <json/json.hpp>
#include <csv/csv.h>

#include "bitmask.hpp"
#include "encoder.hpp"
#include "index.hpp"

using json = nlohmann::json;

struct PartitionHashComparator {
    static size_t hash(std::tuple< Bitmask, int > const & partition_key) {
        size_t seed = std::get<0>(partition_key).hash();
        seed ^=  std::get<1>(partition_key) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        return seed;
    }
    static bool equal(std::tuple<Bitmask, int> const & left, std::tuple<Bitmask, int> const & right) {
        return (std::get<0>(left) == std::get<0>(right)) && (std::get<1>(left) == std::get<1>(right));
    }
};

typedef boost::dynamic_bitset< unsigned long, tbb::scalable_allocator< unsigned long > > bitset;
typedef std::vector< Bitmask, tbb::scalable_allocator< Bitmask > > partition_type;
typedef tbb::concurrent_hash_map< std::tuple< Bitmask, int >, partition_type, PartitionHashComparator, tbb::scalable_allocator< std::pair< std::tuple< Bitmask, int >, partition_type > > > partition_table;
typedef std::unordered_map<unsigned int, float, std::hash<unsigned int>, std::equal_to<unsigned int>, tbb::scalable_allocator< std::pair<const unsigned int, float> > > similarity_index_type;

class Dataset {
public:
    Dataset(void);
    Dataset(std::istream & data_source, unsigned int precision, float regularization, bool verbose);
    ~Dataset(void);

    void load(std::vector< bitset > const & rows, std::tuple< unsigned int, unsigned int > const & split, float const regularization);
    void initialize_kernel(unsigned int platform_index, unsigned int device_index);
    void initialize_similarity_index(float similarity_threshold);

    int const size(void) const; // The true sample size of the dataset
    int const height(void) const; // The physical number of rows needed to represent the dataset
    int const width(void) const; // The number of (binary) source features used to represent the dataset
    int const depth(void) const; // The number of unique target values in the dataset

    // Computes the < lowerbound, upperbound, support, stump-objective > of a subset
    std::tuple< float, float, float, float > impurity(void) const;
    std::tuple< float, float, float, float > impurity(Bitmask const & indicator) const;

    // Computes the bisection of a subset of the data
    partition_type partition(int const feature_index);
    partition_type partition(Bitmask const & indicator, int const feature_index);

    // Compute the properties of a leaf < Prediction, Loss, Complexity >
    std::tuple< std::string, float > leaf(void) const;
    std::tuple< std::string, float > leaf(Bitmask const & indicator) const;

    // Compute the lowerbound of the loss
    float loss_lowerbound(void) const;
    float loss_lowerbound(Bitmask const & indicator) const;

    // Compute the sum of weighted frequency for the k-th label
    float sum(unsigned int k) const;
    float sum(Bitmask const & indicator, unsigned int k) const;

    // Compute the sum of a vector over the indicated subset
    float support(void) const;
    float support(Bitmask const & indicator) const;

    // Computes the entropy of a subset of the data
    float const entropy(Bitmask const & indicator) const;

    // Represents the base similarity index of the full dataset 
    // This distances in this matrix get smaller and smaller for smaller subsets, so we might want a way to calculate that if it's not too expensive?
    std::unordered_map<unsigned int, similarity_index_type, std::hash<unsigned int>, std::equal_to<unsigned int>, tbb::scalable_allocator<std::pair<const unsigned int, similarity_index_type>>> similarity_index;

    // The encoder used in converting between non-binary and binary
    Encoder encoder;

    // Compute total support
    float distance(Bitmask const & indicator, int feature_a, int feature_b);

private:
    // unordered map for pairs of features?
    // total support vector, all label indices summed element wise
    bool verbose;
    std::tuple< int, int, int > shape;
    int _size;
    float _regularization;

    std::vector< std::tuple< Bitmask, int > > labels;
    std::vector< std::tuple< Bitmask, std::vector< float > > > rows;

    std::vector< Bitmask > columns;
    std::vector< Index > label_indices;
    Index total_support;
    Index loss_index;

    std::map< std::pair< int, int >, Index > distance_map;

    partition_table partition_cache;    
    std::vector< Bitmask, tbb::scalable_allocator< Bitmask > > _partition(Bitmask const & indicator, int const feature_index);

    similarity_index_type const & similar_features(unsigned int j) const;

    std::tuple< float, float, float , float > _impurity(Bitmask const & indicator) const;
    std::tuple< float, float, float , float > _kernel_impurity(Bitmask const & indicator) const;
    
    bool hardware_acceleration = false;
    #ifdef INCLUDE_OPENCL
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    std::vector< cl::Buffer > buffers;
    size_t work_group_size;
    size_t work_group_count;
    size_t work_item_count;
    #endif
};

#endif