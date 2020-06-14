#ifndef OPTIMIZER_H
#define OPTIMIZER_H

// Invocation Priorities
#define DELEGATION_PRIORITY 0
#define INFORMATION_PRIORITY 1
#define RESCOPE_PRIORITY 0

#include <iostream>
#include <fstream>
#include <sstream>

#include <queue>

#include <chrono>
#include <unordered_map>
#include <boost/dynamic_bitset.hpp>
#include <tbb/tick_count.h>
#include <tbb/scalable_allocator.h>
#include <json/json.hpp>


#include "dataset.hpp"
#include "model.hpp"
#include "graph.hpp"
#include "queue.hpp"

using json = nlohmann::json;

typedef boost::dynamic_bitset< unsigned long, tbb::scalable_allocator< unsigned long > > bitset;

class Optimizer {
public:
    Optimizer(void);
    Optimizer(Dataset & dataset, json const & configuration);
    ~Optimizer(void);

    std::tuple< float, float> objective_boundary(void) const; // Returns the optimality gap's boundaries
    float const uncertainty(void); // Determine the size of the optimality gap
    bool const complete(void); // Determine if the overall problem has reached the target optimality gap

    bool const tick(int const id = 0); // Attempt a tick (Profiler is triggered here)
    float const elapsed(void); // Determine the elapsed time
    bool const timeout(void); // Determine if the time limit has been reached

    void iterate(int const id = 0); // Main iteration kernel which moves the optimization state forward

    std::unordered_set< Model > models(unsigned int limit = 0); // Returns a (non-exhaustive) set of possibly optimal models given the current information

    void diagnose_non_convergence(void); // Prints the remaining dependency that needs to be resolved in order to reach global optimality
    void diagnose_false_convergence(void); // Prints the remaining dependency that needs to be resolved in order to reach global optimality

    unsigned int size(void);
    
private:
    // Configuration Members
    json _configuration;
    float regularization;
    float uncertainty_tolerance;
    float time_limit;
    unsigned int output_limit;

    float optimism;
    float equity;
    unsigned int sample_depth;
    float similarity_threshold;

    unsigned int workers;
    unsigned long ticks = 0;
    unsigned long tick_duration = 1;

    std::string profile_output = "";
    std::string timing_output = "";

    // Optimization State
    Encoder encoder;
    Dataset dataset;
    Graph graph;
    Queue queue;

    // Timing State
    tbb::tick_count start_time;
    tbb::tick_count last_tick;

    json const & configuration(void) const;

    // Result Extraction
    std::unordered_set< Model > models(Key const & key, Task & task, unsigned int limit = 0);

    void diagnose_non_convergent_task(Key const &key);
    void diagnose_falsely_convergent_task(Key const &key);

    // Graph update kernel
    // This is the function ran repeatedly on each node
    void execute(Key const & key);

    // Helpers to reduce the complexity of the graph update kernel
    Task new_task(Key const & key, Bitmask const & sensitivity, similarity_index_table_type const & parent_similarity_index);
    float const sample(Key const & key, Bitmask const & sensitivity, int const depth);
    void async_call(Key const & key, float const primary_priority = 0, float const secondary_priority = 0, float const tertiary_priority = 0);
    void async_return(Key const & key, float const primary_priority = 0, float const secondary_priority = 0, float const tertiary_priority = 0);

};

#endif